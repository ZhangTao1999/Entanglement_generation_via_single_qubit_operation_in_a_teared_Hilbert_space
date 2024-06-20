# Entanglement_generation_via_single_qubit_operation_in_a_teared_Hilbert_space
In this repository, we give show relevant code & data for **Entanglement_generation_via_single_qubit_operation_in_a_teared_Hilbert_space**, including:
1. All the optimized data for reproduction of the results demonstrated in the main text.
2. Necessary code to verify the results in the main text.

We have used vigorous Lindblad and input-output formalism to check the validity of our scheme at relatively small atom number (N=20), along with the intra-cavity photonic states.
The illustration for W state and GHZ state generation can be found below.
One can see the difference between the evolution of the atom-photon states without and with the application of $H_{AC}$, where in the latter case the application of strong dispersive coupling renders the evolution of atomic states bounded in a Hilbert subspace, thus "tearing" the total Hilbert space.

Genration of W state for 10 and 20 atoms with barrier created by $H_{AC}$, respectively:

![W_state_Lindblad_10_atom](https://github.com/ZhangTao1999/Entanglement_generation_via_single_qubit_operation_in_a_teared_Hilbert_space/assets/96274358/2e8189f5-046c-45f1-a29f-fcac5aefa23e)
![W_state_Lindblad_20_atom](https://github.com/ZhangTao1999/Entanglement_generation_via_single_qubit_operation_in_a_teared_Hilbert_space/assets/96274358/2ae32138-48c0-44f8-adb2-573e152e8417)

By stopping at certain time spot in the gif, we create a W state with high fidelity. We deliberately extended the evolution time to show the effectiveness of the barrier created by $H_{AC}$.



Generation of GHZ state with a relatively low $H_{AC}$:

![GHZ_Lindblad_WithoutPumping](https://github.com/ZhangTao1999/Entanglement_generation_via_single_qubit_operation_in_a_teared_Hilbert_space/assets/96274358/4c7ae8b9-8dbb-4d04-8897-9616d7e00f3f)

For comparison, without barrier:

![GHZ_Lindblad_WithoutPumping](https://github.com/ZhangTao1999/Entanglement_generation_via_single_qubit_operation_in_a_teared_Hilbert_space/assets/96274358/0cbb2c87-0449-4c64-9ae8-b79538790b7f)

