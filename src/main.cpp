#include <Arduino.h>
#include "sin_predictor.h" 
#include "TensorFlowLite.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

#define INPUT_BUFFER_SIZE 64
#define OUTPUT_BUFFER_SIZE 64
#define INT_ARRAY_SIZE 8
#define REQUIRED_INPUT_SIZE 7

// put function declarations here:
int string_to_array(char *in_str, int *int_array);
void print_int_array(int *int_array, int array_len);
int sum_array(int *int_array, int array_len);
void run_tflm_inference(int *int_array);

char received_char = (char)NULL;              
int chars_avail = 0;                    // input present on terminal
char out_str_buff[OUTPUT_BUFFER_SIZE];  // strings to print to terminal
char in_str_buff[INPUT_BUFFER_SIZE];    // stores input from terminal
int input_array[INT_ARRAY_SIZE];        // array of integers input by user

int in_buff_idx=0; // tracks current input location in input buffer
int array_length=0;
int array_sum=0;

// TensorFlow Lite model setup
constexpr int tensor_arena_size = 2 * 1024; // Adjust memory allocation
uint8_t tensor_arena[tensor_arena_size];

const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
tflite::AllOpsResolver resolver;
TfLiteTensor* input_tensor = nullptr;
TfLiteTensor* output_tensor = nullptr;

void setup() {
  // put your setup code here, to run once:
  delay(5000);
  // Arduino does not have a stdout, so printf does not work easily
  // So to print fixed messages (without variables), use 
  // Serial.println() (appends new-line)  or Serial.print() (no added new-line)
  Serial.println("Test Project waking up");
  memset(in_str_buff, (char)0, INPUT_BUFFER_SIZE*sizeof(char)); 
   // Load the TensorFlow Lite model

   model = tflite::GetModel(sin_predictor_tflite);
   if (model->version() != TFLITE_SCHEMA_VERSION) {
     Serial.println("Model schema version mismatch!");
     return;
   }

   static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, tensor_arena_size);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("Failed to allocate TFLM tensors!");
    return;
  }

  input_tensor = interpreter->input(0);
  output_tensor = interpreter->output(0);
}


void loop() {
  // put your main code here, to run repeatedly:

  // check if characters are avialble on the terminal input
  chars_avail = Serial.available(); 
  if (chars_avail > 0) {
    received_char = Serial.read(); // get the typed character and 
    Serial.print(received_char);   // echo to the terminal

    in_str_buff[in_buff_idx++] = received_char; // add it to the buffer
    if (received_char == 13) { // 13 decimal = newline character
      // user hit 'enter', so we'll process the line.
      Serial.print("About to process line: ");
      Serial.println(in_str_buff);

      // Process and print out the array
      array_length = string_to_array(in_str_buff, input_array);
      if (array_length != REQUIRED_INPUT_SIZE) {
        Serial.println("Error: Please enter exactly 7 numbers.");
      } else {
        Serial.print("Read in numbers: ");
        print_int_array(input_array, array_length);

        run_tflm_inference(input_array);
      }

      // Now clear the input buffer and reset the index to 0
      memset(in_str_buff, (char)0, INPUT_BUFFER_SIZE*sizeof(char)); 
      in_buff_idx = 0;
    }
    else if (in_buff_idx >= INPUT_BUFFER_SIZE) {
      memset(in_str_buff, (char)0, INPUT_BUFFER_SIZE*sizeof(char)); 
      in_buff_idx = 0;
    }    
  }
}

void run_tflm_inference(int *int_array) {
  Serial.println("Running TensorFlow Lite inference...");

  for (int i = 0; i < REQUIRED_INPUT_SIZE; i++) {
    input_tensor->data.f[i] = static_cast<float>(int_array[i]);
  }

  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("TFLM inference failed!");
    return;
  }

  float prediction = output_tensor->data.f[0];
  Serial.print("Prediction result: ");
  Serial.println(prediction);
}


int string_to_array(char *in_str, int *int_array) {
  int num_integers=0;
  char *token = strtok(in_str, ",");
  
  while (token != NULL) {
    int_array[num_integers++] = atoi(token);
    token = strtok(NULL, ",");
    if (num_integers >= INT_ARRAY_SIZE) {
      break;
    }
  }
  
  return num_integers;
}

void print_int_array(int *int_array, int array_len) {
  int curr_pos = 0; // track where in the output buffer we're writing

  sprintf(out_str_buff, "Integers: [");
  curr_pos = strlen(out_str_buff); // so the next write adds to the end
  for(int i=0;i<array_len;i++) {
    // sprintf returns number of char's written. use it to update current position
    curr_pos += sprintf(out_str_buff+curr_pos, "%d, ", int_array[i]);
  }
  sprintf(out_str_buff+curr_pos, "]\r\n");
  Serial.print(out_str_buff);
}

int sum_array(int *int_array, int array_len) {
  int curr_sum = 0; // running sum of the array

  for(int i=0;i<array_len;i++) {
    curr_sum += int_array[i];
  }
  return curr_sum;
}