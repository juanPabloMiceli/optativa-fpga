library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;

use work.perceptron_package.perceptron_input;

entity mnist_model is
port (
  clock_i : in std_logic;
  clr_i : in std_logic;
  enable_i : in std_logic;
  pesos_i : in perceptron_input(((64)*100)+(100*10)-1 downto 0);
  inputs_i : in perceptron_input(63 downto 0);
  bcd_digit_o : out std_logic_vector(3 downto 0)
);
end entity;

architecture logic of mnist_model is
  constant MATRIX_COLUMNS : natural := 3; -- Input + Middle + Output
  constant ROWS_PER_COLUMN : natural := 100;
  constant MATRIX_INPUTS : natural := 64;
  constant MATRIX_OUTPUTS : natural := 10;
  constant INPUT_WEIGHTS: natural := ROWS_PER_COLUMN * MATRIX_INPUTS;
  constant MIDDLE_WEIGHTS: natural := (MATRIX_COLUMNS-3) * (ROWS_PER_COLUMN * ROWS_PER_COLUMN);
  constant OUTPUT_WEIGHTS: natural := ROWS_PER_COLUMN * MATRIX_OUTPUTS;
  constant TOTAL_WEIGHTS: natural := INPUT_WEIGHTS + MIDDLE_WEIGHTS + OUTPUT_WEIGHTS;

  signal neural_network_output : perceptron_input(9 downto 0);
begin

  neural_network : entity work.perceptron_matrix
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
		  outputs_o => neural_network_output
		  );

  output_to_bcd : entity work.output_to_digit
  port map (
    inputs_i => neural_network_output,
    bcd_digit_o => bcd_digit_o
  );
end architecture;
