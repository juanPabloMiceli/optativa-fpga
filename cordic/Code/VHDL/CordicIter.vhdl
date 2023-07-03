library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;

entity cordic_iter is
  generic(
    N     : natural := 16;  --Ancho de la palabra
    SHIFT : natural := 1); --Desplazamiento
  port(
    clk   : in std_logic;
    rst   : in std_logic;
    en_i  : in std_logic;
    xi    : in std_logic_vector (N-1 downto 0);
    yi    : in std_logic_vector (N-1 downto 0);
    zi    : in std_logic_vector (N-1 downto 0);
    ci    : in std_logic_vector (N-1 downto 0);
    dv_o  : out std_logic;
    xip1  : out std_logic_vector (N-1 downto 0);
    yip1  : out std_logic_vector (N-1 downto 0);
    zip1  : out std_logic_vector (N-1 downto 0)
  );
end entity;

architecture structural of cordic_iter is

    signal sxi : std_logic_vector (N-1 downto 0) := (others => '0');
    signal syi : std_logic_vector (N-1 downto 0) := (others => '0');
  begin
    dv_o <= en_i;

    sxi (N-2 downto 0) <= xi (N-1 downto 1);
    syi (N-2 downto 0) <= yi (N-1 downto 1);

    CORDIC: process(clk, rst)
    begin
        if rising_edge(clk) then
            if en_i = '1' then
                if zi(N-1) = '0' then
                    xip1 <= std_logic_vector(signed(xi) - signed(sxi));
                    yip1 <= std_logic_vector(signed(yi) + signed(syi));
                    zip1 <= std_logic_vector(signed(zi) - signed(ci));
                else
                    xip1 <= std_logic_vector(signed(xi) + signed(sxi));
                    yip1 <= std_logic_vector(signed(yi) - signed(syi));
                    zip1 <= std_logic_vector(signed(zi) + signed(ci));
                end if;
            end if;
        end if;
    end process;


end architecture;
