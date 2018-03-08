// Utilities related to measuring system performance,

#include <cstdio>
#include <cstdlib>
#include <cstring>

// Extracts a numeric digit from the line, assuming that a digit exists and the
// line ends in " kB".
int ParseMemInfoLine(char* line) {
  int i = strlen(line);
  const char* p = line;
  while (*p < '0' || *p > '9') p++;
  line[i - 3] = '\0';
  i = atoi(p);
  return i;
}

// Returns the value of a memory-related key from "/proc/self/status" (in KB).
int GetMemoryInfoKB(std::string key) {
  FILE* file = fopen("/proc/self/status", "r");
  int result = -1;
  char line[128];

  while (fgets(line, 128, file) != NULL) {
    if (strncmp(line, key.c_str(), key.length()) == 0) {
      result = ParseMemInfoLine(line);
      break;
    }
  }
  fclose(file);
  return result;
}

// Returns the physical memory usage (in KB) of the current process.
int GetPhysicalKB() { return GetMemoryInfoKB("VmRSS"); }

// Returns the virtual memory usage (in KB) of the current process.
int GetVirtualKB() { return GetMemoryInfoKB("VmSize"); }
