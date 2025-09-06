# split_w1_flat.py
IN_DIM = 784
HIDDEN = 128
with open('W1.mem') as f:
    lines = [ln.strip() for ln in f if ln.strip()]
assert len(lines) >= IN_DIM * HIDDEN, f"need {IN_DIM*HIDDEN} lines"
for neuron in range(HIDDEN):
    start = neuron * IN_DIM
    end = start + IN_DIM
    with open(f"W1_neuron_{neuron}.mem","w") as fo:
        fo.write("\n".join(lines[start:end]) + "\n")
print("W1 split into", HIDDEN, "files.")
