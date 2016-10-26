//
// Created by Ran Xian (xranthoar@gmail.com) on 10/25/16.
//

#include "pgr_camera.h"

PGRCamera::PGRCamera(const string &name, const string &video_uri, int width,
                     int height)
    : Camera(name, video_uri, width, height) {}

bool PGRCamera::Init() { return false; }
bool PGRCamera::OnStop() { return false; }
void PGRCamera::Process() {}
