library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;

use work.perceptron_package.perceptron_input;


entity perceptron_row_test IS
end perceptron_row_test;

architecture behavior of perceptron_row_test is 
    constant INPUTS : natural := 2;
    constant OUTPUTS : natural := 3;
	-- Inputs
	signal pesos_i : perceptron_input((INPUTS*OUTPUTS)-1 downto 0);
	signal peso_0 : std_logic_vector(63 downto 0);
	signal peso_1 : std_logic_vector(63 downto 0);
	signal peso_2 : std_logic_vector(63 downto 0);
	signal peso_3 : std_logic_vector(63 downto 0);
	signal peso_4 : std_logic_vector(63 downto 0);
	signal peso_5 : std_logic_vector(63 downto 0);

	signal inputs_i : perceptron_input(INPUTS-1 downto 0);
	signal input_0 : std_logic_vector(63 downto 0);
	signal input_1 : std_logic_vector(63 downto 0);
	signal clock_i : std_logic := '1';
    signal clr_i : std_logic := '1';
	signal enable_i : std_logic := '1';

	-- Ouputs
	signal outputs_o : perceptron_input(OUTPUTS-1 downto 0);
	signal output_0 : std_logic_vector(63 downto 0);
	signal output_1 : std_logic_vector(63 downto 0);
	signal output_2 : std_logic_vector(63 downto 0);

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

    peso_0 <= b"0000000000000000_0000000000000101_0000000000000000_0000000000000000"; -- 5
    peso_1 <= b"1111111111111111_1111111111111110_0000000000000000_0000000000000000"; -- -2 
    peso_2 <= b"0000000000000000_0000000000000010_0000000000000000_0000000000000000"; -- 2
    peso_3 <= b"0000000000000000_0000000000000001_0000000000000000_0000000000000000"; -- 1
    peso_4 <= b"1111111111111111_1111111111111111_0000000000000000_0000000000000000"; -- -1
    peso_5 <= b"1111111111111111_1111111111111110_0000000000000000_0000000000000000"; -- -2 
	pesos_i(0) <= peso_0;
	pesos_i(1) <= peso_1;
	pesos_i(2) <= peso_2;
	pesos_i(3) <= peso_3;
	pesos_i(4) <= peso_4;
	pesos_i(5) <= peso_5;

    input_0 <= b"0000000000000000_0000000000000010_0000000000000000_0000000000000000"; -- 2
    input_1 <= b"0000000000000000_0000000000000100_0000000000000000_0000000000000000"; -- 4
    inputs_i(0) <= input_0;
    inputs_i(1) <= input_1;

	clr_i <= '1', '0' after 200 ms;
	enable_i <= '0', '1' after 300 ms;

	uut1 : entity work.perceptron_row
    generic map(
        INPUTS => INPUTS,
        OUTPUTS => OUTPUTS
    )
	port map (clock_i => clock_i,
		  clr_i => clr_i,
		  enable_i => enable_i,
		  pesos_i => pesos_i,
		  inputs_i => inputs_i,
		  outputs_o => outputs_o
		  );
    output_0 <= outputs_o(0);
    output_1 <= outputs_o(1);
    output_2 <= outputs_o(2);
end architecture;
