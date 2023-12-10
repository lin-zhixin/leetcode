package interview;

import javax.xml.stream.events.Characters;

public class LY {

//    联影二面
    //     " -1.23"转换为数字
    public void change() {
        String s = " -1.23";
        String news=s.trim();
        System.out.println(s);
        double l = 0, r = 0;
        int sig = news.charAt(0) == '-' ? -1 : 1;
        double spot = 0;
        for (int i = 0; i < news.length(); i++) {
            char c = news.charAt(i);
            if (c == '-') {
                continue;
            }
            if (c == '.') {
                spot = 0.1;
                continue;
            }
            if (spot == 0) {
                l = l * 10 + c - '0';
            } else {
                r = r + (c - '0') * spot;
                spot*=0.1;
            }
        }
//        System.out.println(l+r);
//        System.out.println(r);
        System.out.println(sig*(l + r));
    }

    public static void main(String[] args) {
        LY ly = new LY();
        ly.change();

    }

}
