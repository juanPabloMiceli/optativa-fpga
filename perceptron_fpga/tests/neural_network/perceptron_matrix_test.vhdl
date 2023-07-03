library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;

use work.perceptron_package.perceptron_input;
use work.matrix_weights.GetWeights;


entity perceptron_matrix_test IS
end perceptron_matrix_test;

architecture behavior of perceptron_matrix_test is 
    constant MATRIX_COLUMNS : natural := 4;
    constant ROWS_PER_COLUMN : natural := 3;
    constant MATRIX_INPUTS : natural := 2;
    constant MATRIX_OUTPUTS : natural := 1;
    constant INPUT_WEIGHTS: natural := ROWS_PER_COLUMN * MATRIX_INPUTS;
    constant MIDDLE_WEIGHTS: natural := (MATRIX_COLUMNS-3) * (ROWS_PER_COLUMN * ROWS_PER_COLUMN);
    constant OUTPUT_WEIGHTS: natural := ROWS_PER_COLUMN * MATRIX_OUTPUTS;
    constant TOTAL_WEIGHTS: natural := INPUT_WEIGHTS + MIDDLE_WEIGHTS + OUTPUT_WEIGHTS;
	-- Inputs
	signal pesos_i : perceptron_input(TOTAL_WEIGHTS-1 downto 0);

	signal inputs_i : perceptron_input(MATRIX_INPUTS-1 downto 0);
	signal input_0 : std_logic_vector(63 downto 0);
	signal input_1 : std_logic_vector(63 downto 0);
	signal clock_i : std_logic := '1';
    signal clr_i : std_logic := '1';
	signal enable_i : std_logic := '1';

	-- Ouputs
	signal outputs_o : perceptron_input(MATRIX_OUTPUTS-1 downto 0);
    signal output_0 : std_logic_vector(63 downto 0);

	-- Clock
	constant clock_period : time := 10 ms;

begin
	clock_process : process
	begin
		clock_i <= '0';
		wait for clock_period/2;
		clock_i <= '1';
		wait for clock_period/2;
	end process;

    pesos_i <= GetWeights(0);

    input_0 <= b"0000000000000000_0000000000000011_0000000000000000_0000000000000000"; -- 3
    input_1 <= b"0000000000000000_0000000000000101_0000000000000000_0000000000000000"; -- 5

    inputs_i(0) <= input_0;
    inputs_i(1) <= input_1;

	clr_i <= '1', '0' after 200 ms;
	enable_i <= '0', '1' after 300 ms;

	uut1 : entity work.perceptron_matrix
    generic map(
        COLUMNS => MATRIX_COLUMNS,
        ROWS_PER_COLUMN => ROWS_PER_COLUMN,
        MATRIX_INPUTS => MATRIX_INPUTS,
        MATRIX_OUTPUTS => MATRIX_OUTPUTS
    )
	port map (clock_i => clock_i,
		  clr_i => clr_i,
		  enable_i => enable_i,
		  pesos_in_i => pesos_i(INPUT_WEIGHTS-1 downto 0),
		  pesos_i => pesos_i((MIDDLE_WEIGHTS+INPUT_WEIGHTS)-1 downto INPUT_WEIGHTS),
          pesos_out_i => pesos_i(TOTAL_WEIGHTS-1 downto MIDDLE_WEIGHTS + INPUT_WEIGHTS),
		  inputs_i => inputs_i,
		  outputs_o => outputs_o
		  );
    output_0 <= outputs_o(0);
end architecture;
