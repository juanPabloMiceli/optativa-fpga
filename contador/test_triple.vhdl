library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;

use work.my_package.custom_arr;

entity contador_triple_test IS
end contador_triple_test;

architecture behavior of contador_triple_test is 
	-- Inputs
	signal clock_i : std_logic := '1';
	signal clr_i : std_logic := '0';
	signal enable_i : std_logic := '1';

	-- Ouputs
	signal digits_o : custom_arr(2 downto 0);

	-- Clock
	constant clock_period : time := 100 ms;
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

	uut : entity work.contador_triple
	port map (clock_i => clock_i,
		  clr_i => clr_i,
		  enable_i => enable_i,
		  digits_o => digits_o
		  );
end architecture;
