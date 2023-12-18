package category;

public class MyUtile {

    public static void swap(int[] nums, int a, int b) {
        int t = nums[a];
        nums[a] = nums[b];
        nums[b] = t;
    }
    public static void reverse(int[] nums, int l, int r) {
        while (l < r) {
            swap(nums, l++, r--);
        }
    }

    public static void disMap(boolean[][] a) {
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[0].length; j++) {
                System.out.print(a[i][j] + " ");
            }
            System.out.println();
        }
    }

    public static void dis(int[] a) {
        for (int i = 0; i < a.length; i++) {
            System.out.print(a[i] + " ");
        }
        System.out.println();
    }

    public static void disMap(int[][] a) {
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[i].length; j++) {
                System.out.print("("+i+","+j+") "+a[i][j] + " ");
            }
            System.out.println();
        }
    }
    public static void disMap(long[][] a) {
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[i].length; j++) {
                System.out.print("("+i+","+j+") "+a[i][j] + " ");
            }
            System.out.println();
        }
    }

    public static void printBlank(int count, String s) {
        for (int i = 0; i < count; i++) {
            System.out.print("  ");
        }
        System.out.println(s);
    }

    public static void printReturn(int count, String s1) {
        printBlank(count, "return:" + s1);
//        System.out.print(s1);
    }


}
