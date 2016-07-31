include_directories(${Caffe_INCLUDE_DIRS}
        /usr/local/include)

file(GLOB TX1DNN_SOURCE_FILES "*.h" "*.cpp")

add_library(tx1dnn ${TX1DNN_SOURCE_FILES})

message(${TX1DNN_SOURCE_FILES})

target_link_libraries(tx1dnn
		${GLIB_LIBRARIES}
		${GLIB_GOBJECT_LIBRARIES}
		${OpenCV_LIBRARIES}
		${GLOG_LIBRARIES}
		${GSTREAMER_LIBRARIES}
		${GSTREAMER_APP_LIBRARIES}
		${Caffe_LIBRARIES}
		)

if (Cnmem_FOUND)
	target_link_libraries(tx1dnn ${Cnmem_LIBRARIES})
endif()

if (ON_TEGRA)
	target_link_libraries(tx1dnn ${GIE_LIBRARIES})
endif()

target_include_directories(tx1dnn PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Set Caffe compile flags
function(add_caffe_cflags TARGET)
	SET(CAFFE_CFLAGS "-MMD -MP -pthread -fPIC -O2 -DUSE_OPENCV")

	if (ON_TEGRA)
		SET(CAFFE_CFLAGS "${CAFFE_CFLAGS} -DUSE_CNMEN -DUSE_CUDNN")
	endif()

	get_target_property(TEMP ${TARGET} COMPILE_FLAGS)
	if(TEMP STREQUAL "TEMP-NOTFOUND")
		SET(TEMP "") # set to empty string
	endif()
	# append our values
	SET(TEMP "${CAFFE_CFLAGS} ${TEMP}" )
	set_target_properties(${TARGET} PROPERTIES COMPILE_FLAGS ${TEMP} )
endfunction()

add_caffe_cflags(tx1dnn)