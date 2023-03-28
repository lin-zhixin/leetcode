package category;

public class IntUtil {
    public static void dis(int[] list) {
        for (int i = 0; i < list.length; i++) {
            System.out.print(list[i] + " ");
        }
        System.out.println();
    }
    public static void dis(int[][] map) {
        for (int i = 0; i < map.length; i++) {
            System.out.print(i+":");
            dis(map[i]);
        }
        System.out.println();
    }

}
