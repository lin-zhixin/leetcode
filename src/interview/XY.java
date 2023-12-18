package interview;

import category.MyUtile;

import java.lang.reflect.Proxy;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;
import java.util.concurrent.CyclicBarrier;

public class XY {
    public static void disMap(long[][] a) {
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[i].length; j++) {
                System.out.print("(" + i + "," + j + ") " + a[i][j] + " ");
            }
            System.out.println();
        }
    }

//    public static void main(String[] args) {
//        Scanner in = new Scanner(System.in);
//        while (in.hasNextLine()) {
//            int n = in.nextInt();
//            List<Book> list = new ArrayList<>();
//            for (int i = 0; i < n; i++) {
//                String[] book = in.nextLine().split(" ");
//                list.add(new Book(book[0], Integer.parseInt(book[1])));
//            }
//            list.sort((o1, o2) -> o1.p - o2.p);
//            list.forEach(e -> System.out.println(e.name));
//        }
//
//    }

    //    public static void main(String[] args) {
//        Scanner in = new Scanner(System.in);
//        while (in.hasNextInt()) {
//            int n = in.nextInt();
//            int[] arr = new int[n + 1];
//            for (int i = 1; i < n + 1; i++) {
//                arr[i] = in.nextInt();
//            }
//            int k = in.nextInt(), d = in.nextInt();
////            max
//            long[][] f = new long[n + 1][k + 1];
////            min
//            long[][] g = new long[n + 1][k + 1];
//
//            for (int i = 1; i < n + 1; i++) {
//                f[i][1] = arr[i];
//                g[i][1] = arr[i];
//            }
//
//
//            disMap(f);
//            System.out.println();
//
//            disMap(g);
//            System.out.println();
//
//            for (int i = 2; i <= k; i++) {
//                for (int j = i; j <= n; j++) {
//                    long tmax = -99999, tmin = 99999;
//                    for (int l = Math.max(i - 1, j - d); l <= j - 1; l++) {
////                        tmax = Math.max(tmax, Math.max(f[l][k - 1] * arr[l], g[l][k - 1] * arr[l]));
////                        tmin = Math.min(tmin, Math.max(f[l][k - 1] * arr[l], g[l][k - 1] * arr[l]));
//                        if (tmax < Math.max(f[l][i - 1] * arr[j], g[l][i - 1] * arr[j])) {
//                            tmax = Math.max(f[l][i - 1] * arr[j], g[l][i - 1] * arr[j]);
//                        }
//                        if (tmin > Math.min(f[l][i - 1] * arr[j], g[l][i - 1] * arr[j])) {
//                            tmin = Math.min(f[l][i- 1] * arr[j], g[l][i - 1] * arr[j]);
//                        }
//                        f[j][i] = tmax;
//                        g[j][i] = tmin;
//
//                    }
//
//                }
//            }
//            disMap(f);
//            System.out.println();
//
//            disMap(g);
//            System.out.println();
//
//            long res = -99999;
//            for (int i = k; i <= n; i++) {
//                if (res < f[i][k]) {
//                    res = f[i][k];
//                }
//
//            }
//            System.out.println(res);
//
//        }
//    }


        public static void main(String[] args) {












            Scanner in = new Scanner(System.in);
            int n = in.nextInt();
            int[] num = new int[n+1];
            for(int i=1; i<=n; i++){
                num[i] = in.nextInt();
            }
            int K = in.nextInt();
            int D = in.nextInt();
            long[][] dpm = new long[K+1][n+1];//dpm[i][j]表示选中了i个人，以第j个人结尾的能力最大乘积
            long[][] dpn = new long[K+1][n+1];//dpn[i][j]表示选中了i个人，以第j个人结尾的能力最小乘积
            for(int j=1; j<n+1; j++){//初始
                dpm[1][j] = num[j];
                dpn[1][j] = num[j];
            }
            for(int i=1; i<K+1; i++){//初始
                dpm[i][1] = num[1];
                dpn[i][1] = num[1];
            }
            for(int i=2;i<K+1;i++){
                for(int j=2;j<n+1;j++){
                    for(int k=Math.max(1,j-D);k<j;k++){
                        dpm[i][j] = Math.max(dpm[i][j],Math.max(dpm[i-1][k]*num[j],dpn[i-1][k]*num[j]));
                        dpn[i][j] = Math.min(dpn[i][j],Math.min(dpm[i-1][k]*num[j],dpn[i-1][k]*num[j]));
                    }
                }
            }
            long max = Math.max(dpm[K][1],dpn[K][1]);
            for(int j=2;j<n+1;j++){
                max = Math.max(Math.max(dpm[K][j],dpn[K][j]),max);
            }
            System.out.println(max);
        }

}


//class Book {
//    String name;
//    Integer p;
//
//    public Book() {
//    }
//
//    public Book(String name, Integer p) {
//        this.name = name;
//        this.p = p;
//    }
//
//    @Override
//    public String toString() {
//        return "Book{" +
//                "name='" + name + '\'' +
//                ", p=" + p +
//                '}';
//    }
//}



