package category;

public class Numbers {

//    7. 整数反转
    public int reverse(int x) {

        long t;
        long sig = x < 0 ? -1 : 1;
        x = Math.abs(x);
        long p = 0;
        while (x > 0) {
            p *= 10;
            t = x % 10;
            p += t;
            x /= 10;
        }
        long res = sig * p;
        return res > Integer.MAX_VALUE || res < Integer.MIN_VALUE ? 0 : (int) (res);
    }

    public static void main(String[] args) {
        Numbers o = new Numbers();
        System.out.println(o.reverse(1534236469));
    }
}
