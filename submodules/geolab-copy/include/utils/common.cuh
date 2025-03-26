#pragma once
#ifndef COMMON
#define COMMON
#include<sstream>
#include<Eigen/Eigen>

inline void split(std::string s, std::vector<std::string> &results, char delimeter){
	results.clear();
	if(delimeter == ' '){	// it may appear to skip several contiuous space
		std::string temp = "";
		for(int i=0; i<=s.length(); i++){
			if(s[i] == ' ' || s[i] == '\0'){
				if(temp.length())results.push_back(temp);
			}else{
				temp += s[i];
			}
		}
	}else{
		std::stringstream ss(s);
		std::string temp;
		while(getline(ss, temp, delimeter)){
			results.push_back(temp);
		}
	}
}

inline bool startWith(std::string s, std::string start){
	for(int i=0; start[i] != '\0'; i++){
		if(s[i] != start[i])return false;
	}
	return true;
}


template<typename T>
__host__ __device__ inline void swap_value(T &a, T &b){
	T c(a);
	a = b;
	b = c;
}

inline std::string bytes_to_string(size_t bytes) {
	std::array<std::string, 7> suffixes = {{ "B", "KB", "MB", "GB", "TB", "PB", "EB" }};

	double count = (double)bytes;
	uint32_t i = 0;
	for (; i < suffixes.size() && count >= 1024; ++i) {
		count /= 1024;
	}

	std::ostringstream oss;
	oss.precision(3);
	oss << count << " " << suffixes[i];
	return oss.str();
}

#endif