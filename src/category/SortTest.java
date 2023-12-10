package category;

import java.util.*;

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

    //    详细解释各种快排方法：https://blog.csdn.net/Enthusiastic_boy/article/details/122671897
    public int part(int[] list, int l, int r) {
        int mid = list[l];
        while (l < r) {
            while (l < r && list[r--] >= mid) r--;
            list[l] = list[r];
            while (l < r && list[l++] <= mid) l++;
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

    //     快排非递归
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
        Queue<Integer> help = new LinkedList<>();
        int l1 = l, l2 = mid + 1;
        while (l1 <= mid && l2 <= r) {
            help.add(list[l1] <= list[l2] ? list[l1++] : list[l2++]);
        }
        while (l1 <= mid) {
            help.add(list[l1++]);
        }
        while (l2 <= r) {
            help.add(list[l2++]);
        }
        for (int i = l; i <= r; i++) {
            list[i] = help.poll();
        }
//        help.forEach();
    }

    //    int level=0;
    public void mergeSort(int[] list, int l, int r, int level) {
        MyUtile.printBlank(level++, "l=" + l + ",r=" + r);
        if (l >= r) {
            MyUtile.printReturn(level - 1, null);
            return;
        }
        int mid = (l + r) / 2;
        mergeSort(list, l, mid, level);
        mergeSort(list, mid + 1, r, level);
        MyUtile.printReturn(level, "merge");
        merge(list, l, mid, r);
        MyUtile.printBlank(level--, "l=" + l + ",r=" + r);
    }


    //    912. 排序数组 快排
    public int[] sortArray(int[] nums) {
        qsort(nums);
        return nums;
    }

    public void qsort(int[] nums) {
        Stack<Integer> stack = new Stack<>();
        stack.push(nums.length - 1);
        stack.push(0);
        while (!stack.empty()) {
            int l = stack.pop(), r = stack.pop();
            int p = part2(nums, l, r);
            if (p + 1 < r) {
                stack.push(r);
                stack.push(p + 1);
            }
            if (p - 1 > l) {
                stack.push(p - 1);
                stack.push(l);
            }
        }
    }

    public int part2(int[] nums, int l, int r) {
        int tind = l + new Random().nextInt(r - l + 1);
        int t = nums[l];
        nums[l] = nums[tind];
        nums[tind] = t;
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

    //969. 煎饼排序
    public List<Integer> pancakeSort(int[] arr) {

        List<Integer> res = new ArrayList<Integer>();
        pancakeSort(arr, 0, res);
//        MyUtile.dis(arr);
        return res;
    }

    public void pancakeSort(int[] arr, int sorted, List<Integer> res) {
        int n = arr.length, maxind = getMaxIndex(arr, 0, n - 1 - sorted);
        if (sorted == n) {
            return;
        }
        MyUtile.reverse(arr, 0, maxind);
        res.add(maxind + 1);
        MyUtile.reverse(arr, 0, n - 1 - sorted);
        res.add(n - sorted);
        pancakeSort(arr, sorted + 1, res);
    }

    public int getMaxIndex(int[] nums, int l, int r) {
        int max = nums[l];
        int ind = l;
        for (int i = l; i <= r; i++) {
            if (nums[i] > max) {
                ind = i;
                System.out.println(nums[i] + "ind:" + ind);
                max = nums[i];
            }
        }
        return ind;
    }

    //56. 合并区间
    public int[][] merge(int[][] intervals) {

        Arrays.sort(intervals, (o1, o2) -> {
            if (o1[0] == o2[0]) {
                return o2[1] - o1[1];
            } else {
                return o1[0] - o2[0];
            }
        });
        int l = intervals[0][0], r = intervals[0][1];

        Map<Integer, Integer> map = new HashMap<>();
        if (intervals.length == 1) map.put(l, r);

        for (int i = 1; i < intervals.length; i++) {
            if (l <= intervals[i][0] && intervals[i][1] <= r) {
            } else if (r >= intervals[i][0] && r <= intervals[i][1]) {
                r = intervals[i][1];
//                res.remove(res.size() - 1);
            } else {
                if (i == 1) {
                    map.put(l, r);
                }
                l = intervals[i][0];
                r = intervals[i][1];
            }
            map.put(l, r);
        }
//        res.forEach(e -> {
//            System.out.println(e[0] + "," + e[1]);
//        });
//        System.out.println(res);
        List<int[]> res = new ArrayList<>();
        map.forEach((k, v) -> res.add(new int[]{k, v}));

        return res.toArray(new int[res.size()][]);
    }

    //1288. 删除被覆盖区间
    public int removeCoveredIntervals(int[][] intervals) {
        Arrays.sort(intervals, (o1, o2) -> {
            if (o1[0] == o2[0]) {
                return o2[1] - o1[1];
            } else {
                return o1[0] - o2[0];
            }
        });
        int l = intervals[0][0], r = intervals[0][1];

        Map<Integer, Integer> map = new HashMap<>();
        if (intervals.length == 1) map.put(l, r);

        int del = 0;
        for (int i = 1; i < intervals.length; i++) {
            if (l <= intervals[i][0] && intervals[i][1] <= r) {
                del++;
            } else if (r >= intervals[i][0] && r <= intervals[i][1]) {
                r = intervals[i][1];
//                res.remove(res.size() - 1);
            } else {
                if (i == 1) {
                    map.put(l, r);
                }
                l = intervals[i][0];
                r = intervals[i][1];
            }
            map.put(l, r);
        }
//        res.forEach(e -> {
//            System.out.println(e[0] + "," + e[1]);
//        });
//        System.out.println(res);
        List<int[]> res = new ArrayList<>();
        map.forEach((k, v) -> res.add(new int[]{k, v}));

        return intervals.length - del;

    }

    //986. 区间列表的交集
    public int[][] intervalIntersection(int[][] firstList, int[][] secondList) {
        int i = 0;
        int[][] intervals = new int[firstList.length + secondList.length][];
        for (int j = 0; j < firstList.length; j++) {
            intervals[i++] = firstList[j];
        }
        for (int j = 0; j < secondList.length; j++) {
            intervals[i++] = secondList[j];
        }
        Arrays.sort(intervals, (o1, o2) -> {
            if (o1[0] == o2[0]) {
                return o2[1] - o1[1];
            } else {
                return o1[0] - o2[0];
            }
        });
        int l = intervals[0][0], r = intervals[0][1];

        Map<Integer, Integer> map = new HashMap<>();
        if (intervals.length == 1) map.put(l, r);

        int del = 0;
        for (i = 1; i < intervals.length; i++) {
            if (l <= intervals[i][0] && intervals[i][1] <= r) {
                del++;
                map.put(intervals[i][0], intervals[i][1]);

            } else if (r >= intervals[i][0] && r <= intervals[i][1]) {
                r = intervals[i][1];
                map.put(Math.max(l, intervals[i][0]), r);

//                res.remove(res.size() - 1);
            } else {
                if (i == 1) {
                    map.put(l, r);
                }
                l = intervals[i][0];
                r = intervals[i][1];
                map.put(l, r);
            }
        }
//        res.forEach(e -> {
//            System.out.println(e[0] + "," + e[1]);
//        });
//        System.out.println(res);
        List<int[]> res = new ArrayList<>();
        map.forEach((k, v) -> res.add(new int[]{k, v}));

        return res.toArray(new int[res.size()][]);
    }


    public static void main(String[] args) {

        int[] list = new int[]{5, 3, 4, 1, 2, -89, 54, -2};
        SortTest sortTest = new SortTest();
//        sortTest.insertSort(new int[]{5, 3, 4, 1, 2, -89, 54, -2});
//        sortTest.bobbleSort(new int[]{5, 3, 4, 1, 2, -89, 54, -2});
//        System.out.println(Arrays.toString(sortTest.qsort(new int[]{5, 3, 4, 1, 2, -89, 54, -2}, 0, list.length - 1)));
        sortTest.pancakeSort(list);
        int l = 456;
//        sortTest.mergeSort(list, 0, list.length - 1, 0);
//        sortTest.dis(list);
//        System.out.println(l);
//        sortTest.intervalIntersection(new int[][]{{0, 2}, {5, 10}, {13, 23}, {24, 25}}, new int[][]{{1, 5}, {8, 12}, {15, 24}, {25, 26}});


    }
}
