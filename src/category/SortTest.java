package category;

import java.util.LinkedList;
import java.util.Queue;

public class SortTest {

    public void dis(int[] list) {
        for (int i = 0; i < list.length; i++) {
            System.out.print(list[i] + " ");
        }
        System.out.println();
    }

    public void insertSort(int[] list) {
        System.out.println("插入排序前：");
        dis(list);
        int[] tmpList = list;
        int now, j;
        for (int i = 1; i < tmpList.length; i++) {
            now = tmpList[i];
            j = i - 1;
            while (j >= 0 && now < tmpList[j]) {
                tmpList[j + 1] = tmpList[j];
                j--;
            }
            tmpList[j + 1] = now;
        }
        System.out.println("插入排序后：");
        dis(tmpList);
    }

    public void bobbleSort(int[] list) {
        System.out.println("冒泡排序前：");
        dis(list);
        for (int i = 0; i < list.length; i++) {
            boolean flage = false;
            for (int j = 0; j < list.length - i - 1; j++) {
                if (list[j] > list[j + 1]) {
                    int t = list[j];
                    list[j] = list[j + 1];
                    list[j + 1] = t;
                    flage = true;
                }

            }
            if (!flage) {
                break;
            }
        }
        System.out.println("冒泡排序后：");
        dis(list);
    }

    public int part(int[] list, int l, int r) {
        int mid = list[l];
        while (l < r) {
            while (l < r && list[r] > mid) r--;
            list[l] = list[r];
            while (l < r && list[l] < mid) l++;
            list[r] = list[l];
        }
        list[l] = mid;
        return l;
    }

    public int[] qsort(int[] list, int l, int r) {
        if (l < r) {
            int mid = part(list, l, r);
            qsort(list, l, mid - 1);
            qsort(list, mid + 1, r);
        }
        return list;
    }

    public void merge(int[] list, int l, int mid, int r) {
        Queue<Integer> help = new LinkedList<>();
        int l1 = l, l2 = mid + 1;
        while (l1 <= mid && l2 <= r) {
            help.add(list[l1] <= list[l2] ? list[l1++] : list[l2++]);
        }
        while (l1 <= mid ) {
            help.add(list[l1++]);
        }
        while (l2 <= r) {
            help.add(list[l2++]);
        }
        for (int i = l; i <= r; i++) {
            list[i]=help.poll();
        }
//        help.forEach();
    }
//    int level=0;
    public void mergeSort(int[] list,int l,int r,int level){
        MyUtile.printBlank(level++,"l="+l+",r="+r);
        if (l>=r){
            MyUtile.printReturn(level-1,null);
            return;
        }
        int mid=(l+r)/2;
        mergeSort(list, l, mid,level);
        mergeSort(list, mid+1,r,level);
        MyUtile.printReturn(level,"merge");
        merge(list, l, mid,r);
        MyUtile.printBlank(level--,"l="+l+",r="+r);
    }


    public static void main(String[] args) {

        int[] list = new int[]{5, 3, 4, 1, 2, -89, 54, -2};
        SortTest sortTest = new SortTest();
//        sortTest.insertSort(new int[]{5, 3, 4, 1, 2, -89, 54, -2});
//        sortTest.bobbleSort(new int[]{5, 3, 4, 1, 2, -89, 54, -2});
//        System.out.println(Arrays.toString(sortTest.qsort(new int[]{5, 3, 4, 1, 2, -89, 54, -2}, 0, list.length - 1)));

        int l = 456;
        sortTest.mergeSort(list, 0, list.length-1, 0);
        sortTest.dis(list);
//        System.out.println(l);


    }
}
