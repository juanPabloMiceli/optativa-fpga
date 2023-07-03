library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;

package perceptron_package is
    type perceptron_input is array (natural range <>) of std_logic_vector(63 downto 0);
end package;
