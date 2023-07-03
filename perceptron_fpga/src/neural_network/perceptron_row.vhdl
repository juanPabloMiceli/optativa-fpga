library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;

use work.perceptron_package.perceptron_input;

entity perceptron_row is
generic (
    INPUTS: natural := 1;
    OUTPUTS: natural := 1
);
port (
	clock_i : in std_logic;
	clr_i : in std_logic;
	enable_i : in std_logic;
    pesos_i : in perceptron_input((INPUTS * OUTPUTS)-1 downto 0);
    inputs_i : in perceptron_input(INPUTS-1 downto 0);
    outputs_o : out perceptron_input(OUTPUTS-1 downto 0)
);
end entity;

architecture logic of perceptron_row is 
begin
    MY_PROCESS: for i in 0 to OUTPUTS-1 generate
        PERC: entity work.perceptron
        generic map (
            N => INPUTS
        )
        port map(
            peso_i => pesos_i((INPUTS * (i+1)) - 1 downto INPUTS * i),
            input_i => inputs_i,
            clock_i => clock_i,
            clr_i => clr_i,
            enable_i => enable_i,
            output_o => outputs_o(i)
        );
    end generate;
end architecture;
