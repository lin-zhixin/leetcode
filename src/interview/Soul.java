package interview;

public class Soul {

    public int cnt1(int n) {
        int res = 0;
        for (int i = 1; i <= n; i++) {
            res += get1(i);
        }
        return res;
    }

    public int get1(int n) {
        int sum = 0;
        while (n > 0) {
            if (n % 10 == 1) {
                sum++;
            }
            n /= 10;
        }
        return sum;
    }

    public static void main(String[] args) {
        Soul soul = new Soul();
        System.out.println(soul.cnt1(13));
    }
}
