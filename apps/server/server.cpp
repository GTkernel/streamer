/**
 * @brief runner.cpp - The long running process on the device. This process
 * manages the cameras and streams, run DNN on realtime camera frames, push
 * stats and video frames to local storage.
 */

#include "streamer.h"

#define BOOST_SPIRIT_THREADSAFE

#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <simplewebserver/client_http.hpp>
#include <simplewebserver/server_http.hpp>

typedef SimpleWeb::Server<SimpleWeb::HTTP> HttpServer;
typedef SimpleWeb::Client<SimpleWeb::HTTP> HttpClient;
typedef std::shared_ptr<HttpServer::Response> HttpServerResponse;
typedef std::shared_ptr<HttpServer::Request> HttpServerRequest;

using std::cout;
using std::endl;

int main(int argc, char *argv[]) {
  size_t server_thread_num = 1;
  size_t server_port = 15213;
  HttpServer server(server_port, server_thread_num);
  server.resource["^/hello$"]["GET"] = [](HttpServerResponse response,
                                          HttpServerRequest request) {
    cout << "access /hello" << endl;
    string content = "Hello from streamer\n";
    *response << "HTTP/1.1 200 OK\r\nContent-Length: " << content.length()
              << "\r\n\r\n"
              << content;
  };

  std::thread server_thread([&server]() {
    server.start();
  });

  STREAMER_SLEEP(1000);
  cout << "Streamer server started at " << server_port << endl;
  server_thread.join();
}