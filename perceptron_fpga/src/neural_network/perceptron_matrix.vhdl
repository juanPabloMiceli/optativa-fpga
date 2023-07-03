library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;


use work.perceptron_package.perceptron_input;

entity perceptron_matrix is
generic (
    COLUMNS: natural := 1;
    ROWS_PER_COLUMN: natural := 1;
    MATRIX_INPUTS: natural := 1;
    MATRIX_OUTPUTS: natural := 1
);
-- INPUT WEIGHTS = MATRIX_INPUTS * ROWS_PER_COLUMN
-- MIDDLE WEIGTHS = (ROWS_PER_COLUMN * ROWS_PER_COLUMN) * (COLUMNS - 3)
-- OUTPUT WEIGTHS = MATRIX_OUTPUTS * ROWS_PER_COLUMN
port (
	clock_i : in std_logic;
	clr_i : in std_logic;
	enable_i : in std_logic;
    pesos_out_i : in perceptron_input((MATRIX_OUTPUTS * ROWS_PER_COLUMN)-1 downto 0); -- INPUT WEIGHTS
    pesos_in_i : in perceptron_input((MATRIX_INPUTS * ROWS_PER_COLUMN)-1 downto 0); -- INPUT WEIGHTS
    pesos_i : in perceptron_input(((ROWS_PER_COLUMN * ROWS_PER_COLUMN) * (COLUMNS - 3)) - 1 downto 0); -- MIDDLE WEIGHTS
    inputs_i : in perceptron_input(MATRIX_INPUTS-1 downto 0);
    outputs_o : out perceptron_input(MATRIX_OUTPUTS-1 downto 0)
);
end entity;

architecture logic of perceptron_matrix is 
    type my_custom_type is array (COLUMNS-1 downto 0) of perceptron_input(ROWS_PER_COLUMN-1 downto 0);
    signal intercell_wires : my_custom_type;
    constant weight_per_row : natural := (ROWS_PER_COLUMN * ROWS_PER_COLUMN);
begin
    matrix: for i in 0 to COLUMNS-2 generate
        first_column: if i = 0 generate
            first_col: entity work.perceptron_row
            generic map (
                INPUTS => MATRIX_INPUTS,
                OUTPUTS => ROWS_PER_COLUMN
            )
            port map(
                clock_i => clock_i,
                clr_i => clr_i,
                enable_i => enable_i,
                pesos_i => pesos_in_i,
                inputs_i => inputs_i,
                outputs_o => intercell_wires(i)
            );
        end generate first_column;
        middle_columns: if (i > 0) and (i < COLUMNS-2) generate
            middle_cols: entity work.perceptron_row
            generic map (
                INPUTS => ROWS_PER_COLUMN,
                OUTPUTS => ROWS_PER_COLUMN
            )
            port map(
                clock_i => clock_i,
                clr_i => clr_i,
                enable_i => enable_i,
                pesos_i => pesos_i((weight_per_row*i)-1 downto (weight_per_row * (i-1))),
                inputs_i => intercell_wires(i-1),
                outputs_o => intercell_wires(i)
            );
        end generate middle_columns;
        last_column: if i = COLUMNS-2 generate
            last_col: entity work.perceptron_row
            generic map (
                INPUTS => ROWS_PER_COLUMN,
                OUTPUTS => MATRIX_OUTPUTS
            )
            port map(
                clock_i => clock_i,
                clr_i => clr_i,
                enable_i => enable_i,
                pesos_i => pesos_out_i,
                inputs_i => intercell_wires(i-1),
                outputs_o => outputs_o
            );
        end generate last_column;
    end generate matrix;
end architecture;
