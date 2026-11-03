idf_component_register(
    SRCS main.cpp esp_micro_profiler.cpp benchmark_conv.cpp benchmark_slice.cpp
    PRIV_REQUIRES spi_flash driver esp_timer espressif__esp-nn
    INCLUDE_DIRS ".")

    # Reduce the level of paranoia to be able to compile sources
target_compile_options(${COMPONENT_LIB} PRIVATE
    -Wno-maybe-uninitialized
    -Wno-missing-field-initializers
    -Wno-error=sign-compare
    -Wno-error=double-promotion
    -Wno-type-limits)
