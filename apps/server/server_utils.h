//
// Created by Ran Xian (xranthoar@gmail.com) on 10/31/16.
//

#ifndef STREAMER_SERVER_UTILS_H
#define STREAMER_SERVER_UTILS_H

#include "streamer.h"

#include <sys/stat.h>
#include <boost/filesystem.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <simplewebserver/server_http.hpp>

#define BOOST_SPIRIT_THREADSAFE

typedef SimpleWeb::Server<SimpleWeb::HTTP> HttpServer;
typedef std::shared_ptr<HttpServer::Response> HttpServerResponse;
typedef std::shared_ptr<HttpServer::Request> HttpServerRequest;

namespace pt = boost::property_tree;
namespace fs = boost::filesystem;

string CameraToJson(Camera* camera, pt::ptree& root) {
  std::ostringstream ss;

  root.put("name", camera->GetName());
  root.put("video_uri", camera->GetVideoURI());
  root.put("width", camera->GetWidth());
  root.put("height", camera->GetHeight());
  root.put("started", camera->IsStarted());

  if (camera->IsStarted()) {
    root.put("exposure", camera->GetExposure());
    root.put("sharpness", camera->GetSharpness());
    root.put("camera_mode", (int)camera->GetMode());
  }

  pt::write_json(ss, root);

  return ss.str();
}

string ListToJson(const string& list_name,
                  const std::vector<pt::ptree>& pt_list) {
  pt::ptree root;
  std::ostringstream ss;

  pt::ptree list_node;

  for (const auto& child : pt_list) {
    list_node.push_back({"", child});
  }

  root.add_child(list_name, list_node);
  pt::write_json(ss, root);

  return ss.str();
}

string DirectoryToJson(const string& dir_path) {
  pt::ptree root;

  std::ostringstream ss;

  pt::ptree files_node;

  for (const auto& file : fs::directory_iterator(dir_path)) {
    if (!fs::is_directory(file.status())) {
      string filename = file.path().filename().string();
      string filepath = dir_path + "/" + filename;
      pt::ptree file_node;

      file_node.put("path", filepath);
      file_node.put("size", GetFileSize(filepath));

      // Get file creation time
      struct stat stat_buf;
      CHECK(stat(filepath.c_str(), &stat_buf) == 0);
      const struct tm* time = localtime(&stat_buf.st_mtime);
      char timestr[128];
      strftime(timestr, sizeof(timestr), "%Y-%m-%d+%H:%M:%S", time);
      file_node.put("created_at", string(timestr));
      files_node.push_back({"", file_node});
    }
  }

  root.add_child("files", files_node);
  pt::write_json(ss, root);

  return ss.str();
}

#define RN "\r\n"
#define RN2 "\r\n\r\n"

void Send200Response(HttpServerResponse res, const string& content) {
  *res << "HTTP/1.1 200 OK" << RN << "Content-Length: " << content.length()
       << RN2 << content;
}

void SendResponseSuccess(HttpServerResponse res) {
  pt::ptree doc;
  doc.put("result", "success");
  std::ostringstream ss;
  pt::write_json(ss, doc);

  Send200Response(res, ss.str());
}

void Send400Response(HttpServerResponse res, const string& content) {
  *res << "HTTP/1.1 400 Bad Request" << RN
       << "Content-Length: " << content.length() << RN2 << content;
}

void SendBytes(HttpServer& server, HttpServerResponse res, const char* buf,
               size_t total, const string& content_type) {
  *res << "HTTP/1.1 200 OK" << RN << "Content-Type: " << content_type << RN
       << "Content-Length: " << total << RN2;

  // TODO: Consider send in chunk
  res->write(buf, total);
  server.send(res, [](const boost::system::error_code& ec) {
    if (ec != nullptr) {
      LOG(ERROR) << "Can't send buffer";
    }
  });
}

void RecursiveSend(HttpServer& server, HttpServerResponse res,
                   const std::shared_ptr<std::ifstream>& ifs) {
  const size_t BYTES_PER_TRANSFER = (1 << 16);
  std::vector<char> buf(BYTES_PER_TRANSFER);
  size_t read_length;
  if ((read_length = (size_t)ifs->read(buf.data(), buf.size()).gcount()) > 0) {
    res->write(&buf[0], read_length);
    if (read_length == buf.size()) {
      server.send(res,
                  [&server, res, ifs](const boost::system::error_code& ec) {
                    if (!ec)
                      RecursiveSend(server, res, ifs);
                    else
                      LOG(INFO) << "Connection interrupted";
                  });
    }
  }
}

void SendFile(HttpServer& server, HttpServerResponse res,
              const string& filepath, const string& content_type) {
  CHECK(FileExists(filepath)) << "File not found: " << filepath;

  auto ifs = std::make_shared<std::ifstream>();
  ifs->open(filepath, std::ifstream::in | std::ios::binary);

  ifs->seekg(0, std::ios::end);
  auto total = ifs->tellg();
  ifs->seekg(0, std::ios::beg);

  *res << "HTTP/1.1 200 OK" << RN << "Content-Type: " << content_type << RN
       << "Content-Disposition: attachment; filename="
       << boost::filesystem::path(filepath).filename() << RN
       << "Content-Length: " << total << RN2;

  RecursiveSend(server, res, ifs);

  LOG(INFO) << "Sent file: " << filepath;
}

#endif  // STREAMER_SERVER_UTILS_H
