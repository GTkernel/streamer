#include <gst/gst.h>
#include <gst/app/gstappsink.h>
#include <stdlib.h>
#include <thread>
#include <mutex>
#include <iostream>
#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

std::deque<Mat> frameQueue;
std::mutex frameQueueMutex;

GstFlowReturn
new_preroll(GstAppSink *appsink, gpointer data) {
    g_print ("Got preroll!\n");
    return GST_FLOW_OK;
}

GstFlowReturn
new_sample(GstAppSink *appsink, gpointer data) {
    static int framecount = 0;
    framecount++;

    GstSample *sample = gst_app_sink_pull_sample(appsink);
    GstCaps *caps = gst_sample_get_caps(sample);
    GstBuffer *buffer = gst_sample_get_buffer(sample);
    const GstStructure *info = gst_sample_get_info(sample);

    // ---- Read frame and convert to opencv format ---------------

    GstMapInfo map;
    gst_buffer_map (buffer, &map, GST_MAP_READ);

    // convert gstreamer data to OpenCV Mat, you could actually
    // resolve height / width from caps...
    Mat frame(Size(640, 480), CV_8UC3, (char*)map.data, Mat::AUTO_STEP);
    int frameSize = map.size;

    frameQueueMutex.lock();
    frameQueue.push_back(frame);
    frameQueueMutex.unlock();

    gst_buffer_unmap(buffer, &map);

    // ------------------------------------------------------------

    // print dot every 30 frames
    if (framecount%30 == 0) {
        g_print (".");
    }

    // show caps on first frame
    if (framecount == 1) {
        g_print ("CAPS: %s\n", gst_caps_to_string(caps));
    }

    gst_sample_unref (sample);
    return GST_FLOW_OK;
}

static gboolean
my_bus_callback (GstBus *bus, GstMessage *message, gpointer data) {
    g_print ("Got %s message\n", GST_MESSAGE_TYPE_NAME (message));
    switch (GST_MESSAGE_TYPE (message)) {
        case GST_MESSAGE_ERROR: {
            GError *err;
            gchar *debug;

            gst_message_parse_error (message, &err, &debug);
            g_print ("Error: %s\n", err->message);
            g_error_free (err);
            g_free (debug);
            break;
        }
        case GST_MESSAGE_EOS:
            /* end-of-stream */
            break;
        default:
            /* unhandled message */
            break;
    }
    /* we want to be notified again the next time there is a message
     * on the bus, so returning TRUE (FALSE means we want to stop watching
     * for messages on the bus and our callback should not be called again)
     */
    return TRUE;
}

int
main (int argc, char *argv[])
{
    GError *error = NULL;

    gst_init (&argc, &argv);

    gchar *descr = g_strdup(
            "rtspsrc location=\"rtsp://xxx@camera1/cam/realmonitor?channel=1&subtype=1\" "
                    "! rtph264depay ! h264parse ! omxh264dec ! videoconvert ! capsfilter caps=video/x-raw,format=(string)BGR "
                    "! appsink name=sink sync=true"
    );
    GstElement *pipeline = gst_parse_launch (descr, &error);

    if (error != NULL) {
        g_print ("could not construct pipeline: %s\n", error->message);
        g_error_free (error);
        exit (-1);
    }

    /* get sink */
    GstElement *sink = gst_bin_get_by_name (GST_BIN (pipeline), "sink");

    gst_app_sink_set_emit_signals((GstAppSink*)sink, true);
    gst_app_sink_set_drop((GstAppSink*)sink, true);
    gst_app_sink_set_max_buffers((GstAppSink*)sink, 1);
    GstAppSinkCallbacks callbacks = { NULL, new_preroll, new_sample };
    gst_app_sink_set_callbacks (GST_APP_SINK(sink), &callbacks, NULL, NULL);

    GstBus *bus;
    guint bus_watch_id;
    bus = gst_pipeline_get_bus (GST_PIPELINE (pipeline));
    bus_watch_id = gst_bus_add_watch (bus, my_bus_callback, NULL);
    gst_object_unref (bus);

    gst_element_set_state (GST_ELEMENT (pipeline), GST_STATE_PLAYING);

    while(1) {
        g_main_iteration(false);

        cv::namedWindow("frame");
        if (frameQueue.size() > 0) {
            Mat frame = frameQueue.front();
            cout << "Size " << frame.size[0] << " " << frame.size[1] << endl;
            cv::imwrite("test.jpeg", frame);
            cv::imshow("frame", frame);
            cv::waitKey(30);
            frameQueue.clear();
        }
    }

    gst_element_set_state (GST_ELEMENT (pipeline), GST_STATE_NULL);
    gst_object_unref (GST_OBJECT (pipeline));

    return 0;
}