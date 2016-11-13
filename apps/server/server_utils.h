//
// Created by Ran Xian (xranthoar@gmail.com) on 10/31/16.
//

#ifndef STREAMER_SERVER_UTILS_H
#define STREAMER_SERVER_UTILS_H

#include "streamer.h"

#include <boost/filesystem.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <json/json.hpp>
#include <simplewebserver/server_http.hpp>

#define BOOST_SPIRIT_THREADSAFE

typedef SimpleWeb::Server<SimpleWeb::HTTP> HttpServer;
typedef std::shared_ptr<HttpServer::Response> HttpServerResponse;
typedef std::shared_ptr<HttpServer::Request> HttpServerRequest;

namespace pt = boost::property_tree;

string CameraToJson(Camera *camera, pt::ptree &root) {
  std::ostringstream ss;

  root.put("name", camera->GetName());
  root.put("video_uri", camera->GetVideoURI());
  root.put("width", camera->GetWidth());
  root.put("height", camera->GetHeight());

#ifdef USE_PTGRAY
  if (camera->GetCameraType() == CAMERA_TYPE_PTGRAY) {
    auto ptgray_camera = dynamic_cast<PGRCamera *>(camera);
    root.put("exposure", ptgray_camera->GetExposure());
    root.put("sharpness", ptgray_camera->GetSharpness());
    root.put("camera_mode", (int)ptgray_camera->GetMode());
  }
#endif

  pt::write_json(ss, root);

  return ss.str();
}

string ListToJson(const string &list_name,
                  const std::vector<pt::ptree> &pt_list) {
  pt::ptree root;
  std::ostringstream ss;

  pt::ptree list_node;

  for (auto &child : pt_list) {
    list_node.push_back({"", child});
  }

  root.add_child(list_name, list_node);
  pt::write_json(ss, root);

  return ss.str();
}

#define RN "\r\n"
#define RN2 "\r\n\r\n"

void Send200Response(HttpServerResponse res, const string &content) {
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

void Send400Response(HttpServerResponse res, const string &content) {
  *res << "HTTP/1.1 400 Bad Request" << RN
       << "Content-Length: " << content.length() << RN2 << content;
}

void SendBytes(HttpServer &server, HttpServerResponse res, const char *buf,
               size_t total, const string &content_type) {
  *res << "HTTP/1.1 200 OK" << RN << "Content-Type: " << content_type << RN
       << "Content-Length: " << total << RN2;

  const size_t BYTES_PER_TRANSFER = (1 << 16);
  size_t sent = 0;

  while (sent < total) {
    res->write(buf + sent, BYTES_PER_TRANSFER);
    server.send(res, [](const boost::system::error_code &ec) {
      if (ec != nullptr) {
        LOG(ERROR) << "Can't send buffer";
      }
    });
    sent += BYTES_PER_TRANSFER;
  }
}

void SendFile(HttpServer &server, HttpServerResponse res,
              const string &filepath, const string &content_type) {
  CHECK(FileExists(filepath)) << "File not found: " << filepath;

  std::ifstream ifs(filepath);
  ifs.seekg(0, std::ios::end);
  auto total = ifs.tellg();
  ifs.seekg(0, std::ios::beg);

  *res << "HTTP/1.1 200 OK" << RN << "Content-Type: " << content_type << RN
       << "Content-Disposition: attachment; filename="
       << boost::filesystem::path(filepath).filename() << RN
       << "Content-Length: " << total << RN2;

  const size_t BYTES_PER_TRANSFER = (1 << 16);
  std::vector<char> buf(BYTES_PER_TRANSFER);
  size_t sent = 0;

  while (sent < total) {
    long read = ifs.read(buf.data(), buf.size()).gcount();
    if (read < 0) {
      DLOG(ERROR) << "Can't send file: " << filepath;
      break;
    }
    res->write(buf.data(), read);
    server.send(res, [](const boost::system::error_code &ec) {
      if (ec != nullptr) {
        LOG(ERROR) << "Can't send buffer";
      }
    });
    sent += read;
  }
}

#endif  // STREAMER_SERVER_UTILS_H
