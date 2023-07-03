library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;

entity contador_test IS
end contador_test;

architecture behavior of contador_test is 
	-- Inputs
	signal clock_i : std_logic := '1';
	signal pasamanos_i : std_logic := '1';
	signal clr_i : std_logic := '0';
	signal enable_i : std_logic := '1';

	-- Ouputs
	signal max_number_o : std_logic := '0';
	signal out_o : std_logic_vector(3 downto 0);
	signal pasamanos_o : std_logic;

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

	uut : entity work.contador
	port map (clock_i => clock_i,
		  clr_i => clr_i,
		  pasamanos_i => pasamanos_i,
		  enable_i => enable_i,
		  out_o => out_o,
		  pasamanos_o => pasamanos_o
		  );
end architecture;
