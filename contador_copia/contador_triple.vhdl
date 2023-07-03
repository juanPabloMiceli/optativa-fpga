library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;

package my_package is
	type custom_arr is array (integer range <>) of std_logic_vector (3 downto 0);
end package;

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;

use work.my_package.custom_arr;

entity contador_triple is
port (
	clock_i : in std_logic;
	clr_i : in std_logic;
	enable_i : in std_logic;
	digits_o : out custom_arr(2 downto 0)
);
end entity;

architecture logic of contador_triple is 
	signal current_int : unsigned(3 downto 0);
	signal count_next : std_logic_vector(4 downto 0);
	signal pasamanos : std_logic_vector(4 downto 0);
	constant MAX_NUMBER : natural := natural(9);
begin
	count_next(0) <= enable_i;
	pasamanos(0) <= '1';
	DIGITS: for i in 0 to 2 generate
		CONT: entity work.contador
		port map(
			clock_i => clock_i,
			clr_i => clr_i,
			max_number_o => count_next(i+1),
			enable_i => count_next(i),
			out_o => digits_o(i),
			pasamanos_o => pasamanos(i+1),
			pasamanos_i => pasamanos(i)
			);
	end generate;
end architecture;
