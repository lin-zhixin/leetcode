package category;

import java.util.Arrays;
import java.util.PriorityQueue;

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
            while (l < r && list[r] >= mid) r--;//要考虑到相等的情况
            list[l] = list[r];
            while (l < r && list[l] <= mid) l++;
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

    //215. 数组中的第K个最大元素
    public int findKthLargest(int[] nums, int k) {
        findK(nums,nums.length-k,0, nums.length-1);
        return nums[nums.length-k];
    }

    public void findK(int[] nums, int k, int l, int r) {
        if (l < r) {
            int mid = part(nums, l, r);
            if (mid == k) {
            } else if (mid < k) {
                findK(nums, k, l, mid - 1);
                findK(nums, k, mid + 1, r);
            } else {
                findK(nums, k, l, mid - 1);
            }
        }
    }

    PriorityQueue<Integer> pq = new PriorityQueue<>();


    public static void main(String[] args) {

        int[] list = new int[]{5, 3, 4, 1, 2, -89, 54, -2};
        SortTest sortTest = new SortTest();
//        sortTest.insertSort(new int[]{5, 3, 4, 1, 2, -89, 54, -2});
//        sortTest.bobbleSort(new int[]{5, 3, 4, 1, 2, -89, 54, -2});
//        System.out.println(Arrays.toString(sortTest.qsort(new int[]{5, 3, 4, 1, 2, -89, 54, -2}, 0, list.length - 1)));
        System.out.println(sortTest.findKthLargest(new int[]{3,2,3,1,2,4,5,5,6,4},4));

    }
}
