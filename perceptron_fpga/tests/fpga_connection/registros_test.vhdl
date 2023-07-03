library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;

use work.perceptron_package.perceptron_input;


entity registros_test IS
end registros_test;

architecture behavior of registros_test is 
    constant INPUTS : natural := 64;
	-- Inputs

	signal inputs_i : perceptron_input(1 downto 0);
	signal clock_i : std_logic := '1';
    signal clr_i : std_logic := '1';
	signal enable_i : std_logic := '1';
	signal output_valid : std_logic;


	-- Ouputs
	signal outputs_o : perceptron_input(INPUTS - 1 downto 0);
	signal output_0 : std_logic_vector(63 downto 0);
	signal output_1 : std_logic_vector(63 downto 0);
	signal output_2 : std_logic_vector(63 downto 0);

	-- Clock
	constant clock_period : time := 10 ms;

    signal output_contador : std_logic_vector(63 downto 0);
begin
	clock_process : process
	begin
		clock_i <= '0';
		wait for clock_period/2;
		clock_i <= '1';
		wait for clock_period/2;
	end process;

	clr_i <= '1', '0' after 200 ms;
	enable_i <= '0', '1' after 300 ms;

	contador: entity work.contador
    generic map(MAX_NUMBER => INPUTS - 1)
	port map (clock_i => clock_i,
		  clr_i => clr_i,
          enable_i => enable_i,
          out_o => output_contador
      );

	registros: entity work.registros
    generic map(NEURALNET_INPUT => INPUTS)
	port map (
        clock_i => clock_i,
        clr_i => clr_i,
        input_i => output_contador,
        selector => output_contador,
        write_enable_i => enable_i,
        outputs_o => outputs_o,
        output_valid => output_valid
    );
    output_0 <= outputs_o(0);
    output_1 <= outputs_o(1);
    output_2 <= outputs_o(2);

end architecture;
