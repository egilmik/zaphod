cmake --build --preset linux-gcc
python3 scripts/gen_tune_config.py src/params.h -o config.json
mv config.json ../weather-factory/
mv build/linux-gcc/bin/Zaphod ../weather-factory/tuner/
