library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;
--use work.fixed_generic_pkg_mod.all;

use work.perceptron_package.perceptron_input;


entity perceptron_test IS
end perceptron_test;

architecture behavior of perceptron_test is 
    constant N : natural := 2;
	-- Inputs
	signal peso_1 : perceptron_input(1 downto 0);
	--signal peso_2 : perceptron_input(0 downto 0);
	--signal peso_1_v : std_logic_vector(63 downto 0);
	--signal peso_2_v : std_logic_vector(63 downto 0);

	signal output_1 : std_logic_vector(63 downto 0);
	--signal input_v : std_logic_vector(63 downto 0);
	signal input_in : perceptron_input(1 downto 0);
	--signal input_in2 : perceptron_input(0 downto 0);
	signal clock_i : std_logic := '1';
    signal clr_i : std_logic := '1';
	signal enable_i : std_logic := '1';

	-- Ouputs
	signal output_o : std_logic_vector(63 downto 0);

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

    --peso_1_v <= std_logic_vector(to_signed(8589934592, 64)); --2
    --peso_1_v <= b"0000000000000000_0000000000000010_0000000000000000_0000000000000000";
	--peso_1(0) <= peso_1_v;
    peso_1(0) <= b"0000000000000000_0000000000000000_0001100110011001_1001100110011010"; -- 0.1
    peso_1(1) <= b"1111111111111111_1111111111111111_1110011001100110_0110011001100110"; -- -0.1
    --peso_2_v <= std_logic_vector(to_signed(12884901888, 64)); --3
    --peso_2_v <= b"0000000000000000_0000000000000011_0000000000000000_0000000000000000";
--
    --peso_2(0) <= peso_2_v;
	--103079215104 == 24
    --input_v <= std_logic_vector(to_signed(17179869184, 64)); --4
    --input_v <= b"0000000000000000_0000000000000100_0000000000000000_0000000000000000";
    --input_in(0) <= input_v;
    input_in(0) <= b"0000000000000000_0000000000000000_0001100110011001_1001100110011010"; -- 0.1
    input_in(1) <= b"0000000000000000_0000000000000000_0000010100011110_1011100001010010"; -- 0.02
 

	clr_i <= '1', '0' after 200 ms;
	enable_i <= '0', '1' after 300 ms;

	uut1 : entity work.perceptron
    generic map(N => N)
	port map (clock_i => clock_i,
		  clr_i => clr_i,
		  peso_i => peso_1,
		  input_i => input_in,
		  enable_i => enable_i,
		  output_o => output_1
		  );

	--input_in2(0) <= output_1;
--
	--uut2 : entity work.perceptron
    --generic map(N => N)
	--port map (clock_i => clock_i,
		  --clr_i => clr_i,
		  --peso_i => peso_2,
		  --input_i => input_in2,
		  --enable_i => enable_i,
		  --output_o => output_o
		  --);
end architecture;
