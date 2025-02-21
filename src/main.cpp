#include <Arduino.h>
#include <TensorFlowLite.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include <tensorflow/lite/micro/micro_mutable_op_resolver.h>
#include <tensorflow/lite/micro/micro_error_reporter.h>
#include "tensorflow/lite/schema/schema_generated.h"
#include <tensorflow/lite/version.h>
#include "sin_predictor.h" 

#define INPUT_BUFFER_SIZE 64
#define OUTPUT_BUFFER_SIZE 64
#define INT_ARRAY_SIZE 8  // Must be 8 (1 extra for model input format)
#define EXPECTED_INPUT_SIZE 7  // Model expects 7 input values

// TensorFlow Lite components
tflite::MicroErrorReporter micro_error_reporter;
tflite::AllOpsResolver resolver;
constexpr int tensor_arena_size = 2 * 1024;
uint8_t tensor_arena[tensor_arena_size];

const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// Input buffers
char received_char = (char)NULL;
int chars_avail = 0;
char out_str_buff[OUTPUT_BUFFER_SIZE];
char in_str_buff[INPUT_BUFFER_SIZE];
int input_array[INT_ARRAY_SIZE];

int in_buff_idx = 0;
int array_length = 0;
int array_sum = 0;

// Function declarations
int string_to_array(char *in_str, int *int_array);
void print_int_array(int *int_array, int array_len);
int sum_array(int *int_array, int array_len);

void setup() {
    delay(5000);
    Serial.begin(9600);
    // Arduino does not have a stdout, so printf does not work easily
    // So to print fixed messages (without variables), use 
    // Serial.println() (appends new-line)  or Serial.print() (no added new-line)
    Serial.println("Test Project waking up");
    
    // Load the model
    model = tflite::GetModel(sin_predictor_tflite);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.println("Model version mismatch!");
        return;
    }

    interpreter = new tflite::MicroInterpreter(model, resolver, tensor_arena, tensor_arena_size, &micro_error_reporter);
    interpreter->AllocateTensors();

    input = interpreter->input(0);
    output = interpreter->output(0);

    if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("Failed to allocate TFLM tensors!");
    return;
    }
    Serial.println("TFLM Model Initialized.");
}

void loop() {
    // put your main code here, to run repeatedly:

    // check if characters are available on the terminal input
    chars_avail = Serial.available();
    if (chars_avail > 0) {
        received_char = Serial.read(); // get the typed character and 
        Serial.print(received_char); // echo to the terminal
        
        in_str_buff[in_buff_idx++] = received_char; // add it to the buffer
        if (received_char == 13) { // 13 decimal = newline character
            // user hit 'enter', so we'll process the line.
            Serial.println("\nProcessing input...");

            // Process and print out the array
            array_length = string_to_array(in_str_buff, input_array);
            sprintf(out_str_buff, "Read in  %d integers: ", array_length);
            Serial.print(out_str_buff);
            print_int_array(input_array, array_length);
            array_sum = sum_array(input_array, array_length);
            sprintf(out_str_buff, "Sums to %d\r\n", array_sum);
            Serial.print(out_str_buff);

            if (array_length != EXPECTED_INPUT_SIZE) {
                Serial.println("Error: Please enter exactly 7 numbers for model prediction");
            } else {
                // Load input into TensorFlow Lite model
                for (int i = 0; i < EXPECTED_INPUT_SIZE; i++) {
                    input->data.int8[i] = static_cast<int8_t>((input_array[i])*(255.0f/6.0f)-128.0f);
                }
                //Time Variables
                unsigned long t0 = micros();
                Serial.println("time statement");
                unsigned long t1 = micros();            
                interpreter->Invoke(); // Run inference
                unsigned long t2 = micros();
                
                // Get model output
                int8_t prediction = output->data.int8[0];
                prediction = prediction  / 32.0f;
                Serial.print("Model Prediction: ");
                Serial.println(prediction);
                            
                // Measure execution time
                unsigned long t_print = t1 - t0;
                unsigned long t_infer = t2 - t1;
                Serial.print("Printing Time (µs): ");
                Serial.println(t_print);
                Serial.print("Inference Time (µs): ");
                Serial.println(t_infer);
            }

            // Now clear the input buffer and reset the index to 0
            memset(in_str_buff, (char)0, INPUT_BUFFER_SIZE * sizeof(char));
            in_buff_idx = 0;
        } else if (in_buff_idx >= INPUT_BUFFER_SIZE) {
            memset(in_str_buff, (char)0, INPUT_BUFFER_SIZE * sizeof(char));
            in_buff_idx = 0;
        }
    }
}

int string_to_array(char *in_str, int *int_array) {
    int num_integers = 0;
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
    int curr_pos = 0;
    sprintf(out_str_buff, "Integers: [");
    curr_pos = strlen(out_str_buff);

    for (int i = 0; i < array_len; i++) {
        curr_pos += sprintf(out_str_buff + curr_pos, "%d, ", int_array[i]);
    }
    sprintf(out_str_buff + curr_pos, "]\r\n");
    Serial.print(out_str_buff);
}

int sum_array(int *int_array, int array_len) {
    int curr_sum = 0; // running sum of the array
  
    for(int i=0;i<array_len;i++) {
      curr_sum += int_array[i];
    }
    return curr_sum;
  }