import chess.polyglot

with open("polyglot_keys.hpp", "w") as f:
    f.write("const uint64_t PolyGlotRandom[781] = {\n")
    for val in chess.polyglot.POLYGLOT_RANDOM_ARRAY:
        f.write(f"    {val}ULL,\n")
    f.write("};\n")
