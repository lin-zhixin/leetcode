package test2;

import category.MyUtile;
import javafx.util.Pair;

import java.util.*;

public class SortTest {
    public void qsort2(int[] nums) {
//        randomChange(nums);
//        qsort2(nums, 0, nums.length - 1);
//
//        非递归
        Stack<Pair<Integer, Integer>> stack = new Stack<>();
        stack.push(new Pair<>(0, nums.length - 1));
        while (!stack.isEmpty()) {
            Pair<Integer, Integer> t = stack.pop();
            int l = t.getKey(), r = t.getValue();
            int p = part2(nums, l, r);
            if (l < p - 1) {
                stack.push(new Pair<>(l, p - 1));
            }
            if (r > p + 1) {
                stack.push(new Pair<>(p + 1, r));
            }
        }
    }

    public void qsort2(int[] nums, int l, int r) {
        if (l < r) {
            int p = part2(nums, l, r);
            qsort2(nums, l, p - 1);
            qsort2(nums, p + 1, r);
        }
    }

    public int part2(int[] nums, int l, int r) {
        int mid = nums[l];
        while (l < r) {
            while (l < r && nums[r] >= mid) r--;
            nums[l] = nums[r];
            while (l < r && nums[l] <= mid) l++;
            nums[r] = nums[l];
        }
        nums[l] = mid;
        return l;
    }

    public void randomChange(int[] nums) {
        Random random = new Random();
        int n = nums.length;
        for (int i = 0; i < n; i++) {
            int target = i + random.nextInt(n - i);
            int t = nums[target];
            nums[target] = nums[i];
            nums[i] = t;
        }
    }


    public void insertSort(int[] list) {
        for (int i = 1; i < list.length; i++) {
            int now = list[i];
            int j = i - 1;
            while (j >= 0 && now < list[j]) {
                list[j + 1] = list[j];
                j--;
            }
            list[j + 1] = now;
        }
        MyUtile.dis(list);

    }

    public void bobbleSort(int[] list) {
        int n = list.length;
        boolean flage = true;

        for (int i = 0; i < n && flage; i++) {
            flage = false;
            for (int j = 0; j < n - i - 1; j++) {
                if (list[j] > list[j + 1]) {
                    flage = true;
                    int t = list[j];
                    list[j] = list[j + 1];
                    list[j + 1] = t;
                }
            }
        }
        MyUtile.dis(list);
    }

    public int part(int[] list, int l, int r) {
        int mid = list[l];
        while (l < r) {
            while (l < r && list[r] >= mid) r--;
            if (l < r) {
                list[l++] = list[r];
            }
            while (l < r && list[l] <= mid) l++;
            if (l < r) {
                list[r--] = list[l];
            }
        }
        list[l] = mid;
        return l;

    }


    public int[] qsort(int[] list, int l, int r) {
        if (l < r) {
            int p = part(list, l, r);
            qsort(list, l, p - 1);
            qsort(list, p + 1, r);
        }
        return list;

    }

    //     非递归
    public int[] quickNor(int[] list, int l, int r) {
        if (l < r) {
            Stack<Integer> stack = new Stack();
            int p = part(list, l, r);
            if (l < p - 1) {
                stack.push(p - 1);
                stack.push(l);
            }
            if (p + 1 < r) {
                stack.push(r);
                stack.push(p + 1);
            }
            while (!stack.isEmpty()) {
                l = stack.pop();
                r = stack.pop();
                p = part(list, l, r);
                if (l < p - 1) {
                    stack.push(p - 1);
                    stack.push(l);
                }
                if (p + 1 < r) {
                    stack.push(r);
                    stack.push(p + 1);
                }
            }
        }
        return list;
    }

    public void merge(int[] list, int l, int mid, int r) {
        Deque<Integer> res = new LinkedList<>();
        int one = l, two = mid + 1;
        while (one <= mid && two <= r) {
            res.offer(list[one] <= list[two] ? list[one++] : list[two++]);
        }
        while (one <= mid) {
            res.offer(list[one++]);
        }
        while (two <= r) {
            res.offer(list[two++]);
        }
        for (int i = l; i <= r; i++) {
            list[i] = res.poll();
        }
    }

    //    int level=0;
    public void mergeSort(int[] list, int l, int r, int level) {
        if (l < r) {
            int p = l + (r - l) / 2;
            mergeSort(list, l, p, level);
            mergeSort(list, p + 1, r, level);
            merge(list, l, p, r);
        }
    }

    public void mergsort(int[] nums, int l, int r) {
        if (l < r) {
            int p = l + (r - l) / 2;
            mergsort(nums, l, p);
            mergsort(nums, p + 1, r);
            merge2(nums, l, p, r);
        }
    }

    public void merge2(int[] nums, int l, int p, int r) {
        int one = l, two = p + 1;
        Deque<Integer> res = new ArrayDeque<>();
        while (one <= p && two <= r) {
            res.offer(nums[one] <= nums[two] ? nums[one++] : nums[two++]);
        }
        while (one <= p) {
            res.offer(nums[one++]);
        }
        while (two <= r) {
            res.offer(nums[two++]);
        }
        for (int i = l; i <= r; i++) {
            nums[i] = res.poll();
        }
    }


    //    堆排
    public void heapSort(int[] list) {
        buildHeap(list);
        int n = list.length;
        for (int i = n - 1; i >= 0; i--) {
            MyUtile.swap(list, 0, i);
            down(list, 0, i);
        }
        MyUtile.dis(list);
    }

    //    建大根堆
    public void buildHeap(int[] list) {
        int n = list.length;
        for (int i = n / 2 - 1; i >= 0; i--) {
            down(list, i, n);
        }
    }

    //    对now节点进行调整
    public void down(int[] list, int now, int end) {
        int parent = now, child = parent * 2 + 1;
        while (child < end) {
//           大根堆
            if (child + 1 < end && list[child + 1] > list[child]) {
                child++;
            }
            if (list[parent] < list[child]) {
                MyUtile.swap(list, parent, child);
            }
            parent = child;
            child = parent * 2 + 1;
        }
    }

    //    桶排序
    public void buketSort(int[] list) {
        int len = maxLen(list), n = list.length, base = 1;
        List<List<Integer>> buket = new ArrayList<>(10);
        for (int i = 0; i < 10; i++) {
            buket.add(new ArrayList<>());
        }
        for (int i = 0; i < len; i++) {


            for (int j = 0; j < n; j++) {
                int t = (list[j] / base) % 10;
                buket.get(t).add(list[j]);
            }
            base *= 10;
            int ind = 0;
            for (int j = 0; j < 10; j++) {
                for (Integer e : buket.get(j)) {
                    list[ind++] = e;
                }
                buket.get(j).clear();
            }
        }
        MyUtile.dis(list);


    }

    public int maxLen(int[] list) {
        int max = 0;
        for (int i = 0; i < list.length; i++) {
            max = Math.max(max, Integer.toString(list[i]).length());
        }

        return max;

    }

    public static void main(String[] args) {

        int[] list = new int[]{5, 3, 4, 1, 2, 89, 54, 2};
        SortTest sortTest = new SortTest();
        sortTest.mergsort(list, 0, list.length - 1);
//        sortTest.insertSort(new int[]{5, 3, 4, 1, 2, -89, 54, -2});
//        sortTest.bobbleSort(new int[]{5, 3, 4, 1, 2, -89, 54, -2});
//        System.out.println(Arrays.toString(sortTest.quickNor(new int[]{5, 3, 4, 1, 2, -89, 54, 5}, 0, list.length - 1)));

//        sortTest.buketSort(list);
        int l = 456;
//        sortTest.mergeSort(list, 0, list.length - 1, 0);
        MyUtile.dis(list);
//        System.out.println(l);
//        sortTest.intervalIntersection(new int[][]{{0, 2}, {5, 10}, {13, 23}, {24, 25}}, new int[][]{{1, 5}, {8, 12}, {15, 24}, {25, 26}});


    }
}
