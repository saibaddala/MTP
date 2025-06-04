CXX = g++

UTILITIES_DIR = ../utilities

CXXFLAGS = -std=c++17 -pthread -O3 \
  -I$(UTILITIES_DIR)

LDFLAGS = -L$(UTILITIES_DIR)/lib \
  -ltensorflow-lite -ltensorflowlite_c -lflatbuffers -lnetfilter_queue -lpthread -latomic

SRC = main.cpp flow.cpp packet_parser.cpp feature_extraction.cpp flow_prediction.cpp predictor.cpp
OBJ = $(SRC:.cpp=.o)
TARGET = packet_classifier

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CXX) $(OBJ) -o $@ $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ) $(TARGET)
