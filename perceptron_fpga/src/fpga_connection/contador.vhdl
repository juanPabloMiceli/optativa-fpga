library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;

entity contador is
generic (
    MAX_NUMBER : natural := 63
);
port (
	clock_i : in std_logic;
	clr_i : in std_logic;
	enable_i : in std_logic;
	out_o : out std_logic_vector(63 downto 0)
	--max_number_o : out std_logic
);
end entity;

architecture logic of contador is 
	signal current_int : unsigned(63 downto 0);
begin
	NEXT_PROC: process(clock_i)
	begin
		if rising_edge(clock_i) then
			if clr_i = '1' then
				current_int <= (others => '0');
                --max_number_o <= '0';
			end if;
			if enable_i = '1' then
				--max_number_o <= '0';
				current_int <= current_int + 1;
				if current_int = MAX_NUMBER - 1 then
					--max_number_o <= '1';
				end if;
				if current_int = MAX_NUMBER then
					current_int <= (others => '0');
				end if;
			end if;
		end if;
	end process;
	out_o <= std_logic_vector(current_int);
end architecture;
