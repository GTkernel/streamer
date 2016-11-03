//
// Created by Ran Xian (xranthoar@gmail.com) on 10/31/16.
//

#ifndef STREAMER_SERVER_UTILS_H
#define STREAMER_SERVER_UTILS_H

#include "streamer.h"

#include <simplewebserver/server_http.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>

#define BOOST_SPIRIT_THREADSAFE

typedef SimpleWeb::Server<SimpleWeb::HTTP> HttpServer;
typedef std::shared_ptr<HttpServer::Response> HttpServerResponse;
typedef std::shared_ptr<HttpServer::Request> HttpServerRequest;

namespace pt = boost::property_tree;

string CameraToJson(Camera *camera) {
  pt::ptree root;
  std::ostringstream ss;

  root.put("name", camera->GetName());
  root.put("video_uri", camera->GetVideoURI());
  root.put("width", camera->GetWidth());
  root.put("height", camera->GetHeight());

#ifdef USE_PTGRAY
  if (camera->GetType() == CAMERA_TYPE_PTGRAY) {
    auto ptgray_camera = std::dynamic_pointer_cast<PGRCamera>(camera);
    root.put("exposure", ptgray_camera->GetExposure());
    root.put("sharpness", ptgray_camera->GetSharpness());
    root.put("video_mode", (int)ptgray_camera->GetVideoMode());
  }
#endif

  pt::write_json(ss, root);

  return ss.str();
}

string ListToJson(const string &list_name, const std::vector<string> &string_list) {
  pt::ptree root;
  std::ostringstream ss;

  pt::ptree list_node;

  for (auto &str : string_list) {
    pt::ptree str_node;
    str_node.put("", str);
    list_node.push_back({"", str_node});
  }

  root.add_child(list_name, list_node);
  pt::write_json(ss, root);

  return ss.str();
}

void Send200Response(HttpServerResponse res, const string &content) {
  *res << "HTTP/1.1 200 OK\r\n"
       << "Content-Length: " << content.length() << "\r\n\r\n"
       << content;
}

void Send400Response(HttpServerResponse res, const string &content) {
  *res << "HTTP/1.1 400 Bad Request\r\n"
       << "Content-Length: " << content.length() << "\r\n\r\n"
       << content;
}


#endif //STREAMER_SERVER_UTILS_H
