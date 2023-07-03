library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;

use work.perceptron_package.perceptron_input;


entity output_to_digit_test IS
end output_to_digit_test;

architecture behavior of output_to_digit_test is 
	-- Inputs
	signal inputs_i : perceptron_input(9 downto 0);
    signal input_0 : std_logic_vector(63 downto 0);
    signal input_1 : std_logic_vector(63 downto 0);
    signal input_2 : std_logic_vector(63 downto 0);
    signal input_3 : std_logic_vector(63 downto 0);
    signal input_4 : std_logic_vector(63 downto 0);
    signal input_5 : std_logic_vector(63 downto 0);
    signal input_6 : std_logic_vector(63 downto 0);
    signal input_7 : std_logic_vector(63 downto 0);
    signal input_8 : std_logic_vector(63 downto 0);
    signal input_9 : std_logic_vector(63 downto 0);
	signal clock_i : std_logic := '1';
    
	-- Ouputs
    signal output_0 : std_logic_vector(3 downto 0);

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
    input_0 <= std_logic_vector(to_signed(-4, 64));
    input_1 <= std_logic_vector(to_signed(-2, 64));
    input_2 <= std_logic_vector(to_signed(-9, 64));
    input_3 <= std_logic_vector(to_signed(-5, 64));
    input_4 <= std_logic_vector(to_signed(-5, 64));
    input_5 <= std_logic_vector(to_signed(-5, 64));
    input_6 <= std_logic_vector(to_signed(-3, 64));
    input_7 <= std_logic_vector(to_signed(-6, 64));
    input_8 <= std_logic_vector(to_signed(-4, 64));
    input_9 <= std_logic_vector(to_signed(-5, 64));

    inputs_i(0) <= input_0;
    inputs_i(1) <= input_1;
    inputs_i(2) <= input_2;
    inputs_i(3) <= input_3;
    inputs_i(4) <= input_4;
    inputs_i(5) <= input_5;
    inputs_i(6) <= input_6;
    inputs_i(7) <= input_7;
    inputs_i(8) <= input_8;
    inputs_i(9) <= input_9;


	uut1 : entity work.output_to_digit
	port map (
        inputs_i => inputs_i,
        bcd_digit_o => output_0
		  );
end architecture;
