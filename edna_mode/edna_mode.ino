/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <TensorFlowLite.h>

#include "main_functions.h"

#include "audio_provider.h"
#include "command_responder.h"
#include "feature_provider.h"
#include "micro_features_micro_model_settings.h"
#include "micro_features_model.h"
#include "recognize_commands.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

#include "micro_features_micro_features_generator.h"

// Globals, used for compatibility with Arduino-style sketches.
namespace {
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* model_input = nullptr;
FeatureProvider* feature_provider = nullptr;
RecognizeCommands* recognizer = nullptr;
int32_t previous_time = 0;

// Create an area of memory to use for input, output, and intermediate arrays.
// The size of this will depend on the model you're using, and may need to be
// determined by experimentation.
constexpr int kTensorArenaSize = 10 * 1024;
uint8_t tensor_arena[kTensorArenaSize];
int8_t feature_buffer[kFeatureElementCount];
int8_t* model_input_buffer = nullptr;
}  // namespace

// Global constants for control
const int buttonPin = 2;
//int buttonState = 0;
//int toggleState = 0;
uint8_t incomingByte = 0;

bool is_first_run_ = true;

// The name of this function is important for Arduino compatibility.
void setup() {
  // Set up logging. Google style is to avoid globals or statics because of
  // lifetime uncertainty, but since this has a trivial destructor it's okay.
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  model = tflite::GetModel(g_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Model provided is schema version %d not equal "
                         "to supported version %d.",
                         model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  // Pull in only the operation implementations we need.
  // This relies on a complete list of all the ops needed by this graph.
  // An easier approach is to just use the AllOpsResolver, but this will
  // incur some penalty in code space for op implementations that are not
  // needed by this graph.
  //
  // tflite::AllOpsResolver resolver;
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::MicroMutableOpResolver<4> micro_op_resolver(error_reporter);
  if (micro_op_resolver.AddDepthwiseConv2D() != kTfLiteOk) {
    return;
  }
  if (micro_op_resolver.AddFullyConnected() != kTfLiteOk) {
    return;
  }
  if (micro_op_resolver.AddSoftmax() != kTfLiteOk) {
    return;
  }
  if (micro_op_resolver.AddReshape() != kTfLiteOk) {
    return;
  }

  // Build an interpreter to run the model with.
  static tflite::MicroInterpreter static_interpreter(
      model, micro_op_resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
    return;
  }

  // Get information about the memory area to use for the model's input.
  model_input = interpreter->input(0);
  if ((model_input->dims->size != 2) || (model_input->dims->data[0] != 1) ||
      (model_input->dims->data[1] !=
       (kFeatureSliceCount * kFeatureSliceSize)) ||
      (model_input->type != kTfLiteInt8)) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Bad input tensor parameters in model");
    return;
  }
  model_input_buffer = model_input->data.int8;

  // Prepare to access the audio spectrograms from a microphone or other source
  // that will provide the inputs to the neural network.
  // NOLINTNEXTLINE(runtime-global-variables)
  static FeatureProvider static_feature_provider(kFeatureElementCount,
                                                 feature_buffer);
  feature_provider = &static_feature_provider;

  static RecognizeCommands static_recognizer(error_reporter);
  recognizer = &static_recognizer;

  previous_time = 0;

  //Added code
  //pinMode(buttonPin, INPUT);
  Serial.begin(9600);
  while (!Serial);
//  TfLiteStatus beginSampling = InitAudioRecording(error_reporter);
//  if(beginSampling != kTfLiteOk) {
//    Serial.println("InitAudio Failure");
//  }
//  else {
  Serial.println("Setup Done");
//  }
  
}

// The name of this function is important for Arduino compatibility.
void loop() {
  //buttonState = digitalRead(buttonPin);
  //if(buttonState == HIGH && toggleState == 0) {
  //  toggleState = 1;
  
  if(Serial.available() > 0) {
    incomingByte = Serial.parseInt();
    
    if(incomingByte > 64 && Serial.read() == '\n') {
      Serial.println("Run Model");

      // Fetch the spectrogram for the current time.
      const int32_t current_time = LatestAudioTimestamp();
      previous_time = current_time - 1000;
      if(previous_time < 0) {
        previous_time = 0;
      }
      int how_many_new_slices = 0;
      Serial.println(current_time);

  //-----------------------------------------------------------------------------------------------------//
  //-----------------------------------------------------------------------------------------------------//
      
      if (is_first_run_) {
        TfLiteStatus init_status = InitializeMicroFeatures(error_reporter);
        is_first_run_ = false;
      }
      TfLiteStatus initAudio = InitAudioRecording(error_reporter);
      while(!g_new_sample){
        Serial.print("waiting for new sample");
        delay(5);
      }
  
      for (int new_slice = 0; new_slice <= kFeatureSliceCount;
             ++new_slice) {
        const int new_step = new_slice;
//        const int new_step = (0 - kFeatureSliceCount + 1) + new_slice;
        const int32_t slice_start_ms = (new_step * kFeatureSliceStrideMs);
        Serial.print(slice_start_ms);
        int16_t* audio_samples = nullptr;
        int audio_samples_size = 0;
        // TODO(petewarden): Fix bug that leads to non-zero slice_start_msGener
        GetAudioSamples(error_reporter, (slice_start_ms > 0 ? slice_start_ms : 0),
                        kFeatureSliceDurationMs, &audio_samples_size,
                        &audio_samples);
                        
        int8_t* new_slice_data = feature_provider->feature_data_ + (new_slice * kFeatureSliceSize);
        size_t num_samples_read;
        TfLiteStatus generate_status = GenerateMicroFeatures(
            error_reporter, audio_samples, audio_samples_size, kFeatureSliceSize,
            new_slice_data, &num_samples_read);
            
        while(!g_new_sample){
          Serial.print(new_step);
          delay(10);
        }
        Serial.println("new Sample");
      }
      TfLiteStatus endAudio = EndAudioRecording(error_reporter);
      Serial.println("Audio Ended");

  //-----------------------------------------------------------------------------------------------------//
  //-----------------------------------------------------------------------------------------------------//
      
      /*TfLiteStatus feature_status = feature_provider->GetSample_PopulateFeatureData(
          error_reporter);
      if (feature_status != kTfLiteOk) {
        TF_LITE_REPORT_ERROR(error_reporter, "Feature generation failed");
        return;
      }*/
      Serial.println("Samples & Features Acquired");
      // Copy feature buffer to input tensor
      for (int i = 0; i < kFeatureElementCount; i++) {
        Serial.print(feature_buffer[i]);
        model_input_buffer[i] = feature_buffer[i];        
      }
      Serial.println("data copied");
      // Run the model on the spectrogram input and make sure it succeeds.
      TfLiteStatus invoke_status = interpreter->Invoke();
      if (invoke_status != kTfLiteOk) {
        TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed");
        return;
      }
      Serial.println("Invoked");
      // Obtain a pointer to the output tensor
      TfLiteTensor* output = interpreter->output(0);
      // Determine whether a command was recognized based on the output of inference
      const char* found_command = nullptr;
      uint8_t score = 0;
      bool is_new_command = false;
      TfLiteStatus process_status = recognizer->ProcessLatestResults(
          output, current_time, &found_command, &score, &is_new_command);
      if (process_status != kTfLiteOk) {
        TF_LITE_REPORT_ERROR(error_reporter,
                             "RecognizeCommands::ProcessLatestResults() failed");
        return;
      }
      // Do something based on the recognized command. The default implementation
      // just prints to the error console, but you should replace this with your
      // own function for a real application.
      RespondToCommand(error_reporter, current_time, found_command, score,
                       is_new_command);
      Serial.println("model done");
    }
    else {
      Serial.println("nope");
    }
  }
  /*} else if (buttonState == LOW) {
    if(toggleState == 1) {
      toggleState = 0;
      Serial.println("Button Off");
      delay(10);
    }
    else {
      Serial.println("other");
      delay(10);
    }
  } */
}
