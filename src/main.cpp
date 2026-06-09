#include <iostream>
#include <string_view>
#include <vector>
#include <optional>
#include <assert.h>
#include <algorithm>
#include <filesystem>

#include "mpsReader.h"
#include "solver.h"

const vector<std::string> INPUT_EXTENSIONS = {".mps", ".QPS"};

bool is_extension_valid(string extension) {
    return (find(INPUT_EXTENSIONS.begin(), INPUT_EXTENSIONS.end(), extension) != INPUT_EXTENSIONS.end());
} 

class CommandParser {
public:
    std::vector<std::string_view> m_tokens;
    std::vector<std::string_view> m_arguments;
    
    CommandParser(const int argc, const char* argv[]) {
        for(int i = 1; i < argc; i++) {
            this->m_tokens.push_back(argv[i]);
        }

        for(size_t i = 0; i < m_tokens.size(); i++) {
            if((i == 0 || !is_option(m_tokens[i-1])) && !is_flag(m_tokens[i]) && !is_option(m_tokens[i])) {
                this->m_arguments.push_back(m_tokens[i]);
            }
        }
    }

    std::optional<std::string_view> get_option(const std::string_view token) const {
        if(!is_option(token))
            return std::nullopt;

        auto it = std::find(m_tokens.begin(), m_tokens.end(), token);
        if(it != m_tokens.end() && ++it != m_tokens.end() && !is_option(*it) && !is_flag(*it)) {
            return *it;
        }
        return std::nullopt;
    } 

    bool has_flag(const std::string_view token) const {
        assert(is_flag(token));
        return std::find(m_tokens.begin(), m_tokens.end(), token) != m_tokens.end();
    }

private:
    bool is_flag(const std::string_view token) const {
        return token == "--help" || token == "-h";
    }

    bool is_option(const std::string_view token) const {
        return token == "-C";
    }
};

void print_help() {
    std::cout << "Advanced Simplex Implementation\n\n"
              << "Usage: simplex [-h | --help] [-C <path>] [<files/directories>]\n\n";
}

int main(const int argc, const char* argv[]) {
    CommandParser parser(argc, argv);

    if(parser.has_flag("--help")) {
        print_help();
    }
    
    std::filesystem::path working_directory = std::filesystem::current_path();

    std::optional<std::string_view> directory_option = parser.get_option("-C");
    if(directory_option.has_value()) {
        std::filesystem::path new_path = std::filesystem::path(directory_option.value());
        if(!std::filesystem::exists(new_path)) {
            std::cerr << "given working directory doesn't exists: " << new_path.string() << std::endl;
            return -1;
        }

        working_directory = new_path;
    }
    
    // iterate through input directories and get valid ones
    std::vector<std::filesystem::path> inputs;
    for(std::string_view input : parser.m_arguments) {
        std::cout << input << std::endl;
        const std::filesystem::path input_path = std::filesystem::path(working_directory / input);
        if(!std::filesystem::exists(input_path)) {
            std::cerr << "invalid input: " << input_path.string() << std::endl;
            continue;
        }

        if(std::filesystem::is_directory(input_path)) {
            for (const auto& entry : std::filesystem::recursive_directory_iterator(input_path)) {
                if (entry.is_regular_file()) {
                    const std::filesystem::path& file_path = entry.path();
                
                    if (is_extension_valid(file_path.extension())) {
                        inputs.push_back(file_path);
                    }
                }
            }
        }
        else {
            if(is_extension_valid(input_path.extension())) {
                inputs.push_back(input_path);
            }
        }
    }

    std::cout << "found " << inputs.size() << " input files" << std::endl;

    // solve given files
    for(std::filesystem::path input : inputs) {
        std::cout << "solving " << input.string() << std::endl;

        mpsReader reader(input.string());
        Solver solver(reader);

        SolutionResult result = solver.solve();

        cout << "Variables: \n" << result.variables << "\nObjective: " << result.objective << endl;
    }
    
    return 0;
}
