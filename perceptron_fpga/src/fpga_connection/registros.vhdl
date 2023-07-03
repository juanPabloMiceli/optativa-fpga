library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;

use work.perceptron_package.perceptron_input;

entity registros is
generic (
    NEURALNET_INPUT : natural := 64
);
port (
	clock_i : in std_logic;
	clr_i : in std_logic;
    input_i : in std_logic_vector(63 downto 0); 
    selector : in std_logic_vector(63 downto 0); -- Esto es log2 de NEURALNET_INPUT
	write_enable_i : in std_logic;
    outputs_o : out perceptron_input(NEURALNET_INPUT - 1 downto 0);
    output_valid : out std_logic
);
end entity;

architecture logic of registros is 
begin
	reg_proc: process(clock_i)
        variable index : integer;
    begin
		if rising_edge(clock_i) then
            if clr_i = '1' then

                clear : for k in 0 to outputs_o'length-1 loop
                    outputs_o(k) <= (others => '0');
                end loop clear;
                output_valid <= '0';
            end if;

			if write_enable_i = '1' then
                index := to_integer(unsigned(selector));
                outputs_o(index) <= input_i;
                if unsigned(selector) = (NEURALNET_INPUT - 1) then
                    output_valid <= '1';
                end if;
			end if;
		end if;
	end process;
end architecture;
