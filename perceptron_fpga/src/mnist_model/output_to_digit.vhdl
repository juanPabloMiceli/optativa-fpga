library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;

use work.perceptron_package.perceptron_input;

entity output_to_digit is
port (
    inputs_i : in perceptron_input(9 downto 0);
    bcd_digit_o : out std_logic_vector(3 downto 0)
);
end entity;

architecture logic of output_to_digit is
    function max_index(input_array : perceptron_input) return integer is
      variable index : integer := 0;
      variable max : signed(63 downto 0) := x"8000_0000_0000_0000";
    begin
      for i in 0 to input_array'length - 1 loop
        if signed(input_array(i)) > max then
          index := i;
          max := signed(input_array(i));
        end if;
      end loop;
      return index;
    end function;
begin
    bcd_digit_o <=  "0000" when max_index(inputs_i) = 0 else
                    "0001" when max_index(inputs_i) = 1 else
                    "0010" when max_index(inputs_i) = 2 else
                    "0011" when max_index(inputs_i) = 3 else
                    "0100" when max_index(inputs_i) = 4 else
                    "0101" when max_index(inputs_i) = 5 else
                    "0110" when max_index(inputs_i) = 6 else
                    "0111" when max_index(inputs_i) = 7 else
                    "1000" when max_index(inputs_i) = 8 else
                    "1001" when max_index(inputs_i) = 9 else
                    "0000";
end architecture;
