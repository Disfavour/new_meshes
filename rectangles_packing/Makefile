CC = g++
TARGET = program
SOURCES = $(wildcard *.cpp)
OBJECTS = $(SOURCES:.cpp=.o)
all: $(TARGET)
$(TARGET): $(OBJECTS)
	$(CC) $(OBJECTS) -o $(TARGET)
%.o: %.cpp
	$(CC) -c $< -o $@
clean:
	rm -f $(OBJECTS) $(TARGET)
.PHONY: all clean