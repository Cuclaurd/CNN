CXX := g++
CXXFLAGS := -std=c++17 -Wall -Wextra -pedantic

INCLUDE_DIR := include
SRC_DIR := src
BUILD_DIR := build

# Detecte les fichiers contenant une fonction main().
# Exemple : fixpoint_test.cpp produit build/fixpoint_test.
APP_SOURCES := $(shell find . -path ./build -prune -o -name '*.cpp' -print | xargs grep -l 'int[[:space:]]*main')
APP_NAMES := $(notdir $(APP_SOURCES:.cpp=))
TARGETS := $(addprefix $(BUILD_DIR)/,$(APP_NAMES))

LIB_SOURCES := $(shell find src -name '*.cpp' -print 2>/dev/null)
LIB_OBJECTS := $(patsubst src/%.cpp,$(BUILD_DIR)/src/%.o,$(LIB_SOURCES))

.PHONY: all test clean

all: $(TARGETS)

$(BUILD_DIR)/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -I$(INCLUDE_DIR) -c $< -o $@

$(BUILD_DIR)/src/%.o: src/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -I$(INCLUDE_DIR) -c $< -o $@

# Associe chaque fichier contenant main() a build/<nom-sans-extension>.
define APP_RULE
$(BUILD_DIR)/$(notdir $(basename $(1))): $(patsubst ./%.cpp,$(BUILD_DIR)/%.o,$(1)) $(LIB_OBJECTS)
	@mkdir -p $$(dir $$@)
	$$(CXX) $$(CXXFLAGS) $$^ -o $$@
endef
$(foreach app,$(APP_SOURCES),$(eval $(call APP_RULE,$(app))))

test: all
	@set -e; \
	for executable in $(TARGETS); do \
		echo "Execution de $$executable"; \
		./$$executable; \
	done

clean:
	rm -rf $(BUILD_DIR)
	find . -name '*.o' -delete
