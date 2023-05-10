package category;

//数字相关题目

import javafx.util.Pair;

import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;
//import org.apache.commons.collections.CollectionUtils;

public class Numbers {
    //    作者：LeetCode-Solution
//    链接：https://leetcode.cn/problems/sqrtx/solution/x-de-ping-fang-gen-by-leetcode-solution/
//    来源：力扣（LeetCode）
//    著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

    //    重要理论知识：一个整数的平方根肯定不会超过它自己的一半 ，因为是要求整数部分 因此可以直接int除，整体的流程就是左右两边取中点，大了右边左移1，小了左边右移1
    public int mySqrt(int x) {
        int left = 0, right = x, ans = -1;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if ((long) mid * mid <= x) {
                ans = mid;
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        return ans;

//        int l = 0, r = x, ans = -1;
//        while (l <= r) {
//            int mid = l + (r - l) / 2;
//            if ((long) mid * mid <= x) {
//                ans = mid;
//                l = mid + 1;
//            } else {
//                r = mid - 1;
//            }
//        }
//        return ans;
    }

    //121. 买卖股票的最佳时机
    public int maxProfit(int[] prices) {
        int min = 9999999, max = 0;
        for (int i = 0; i < prices.length; i++) {
            max = Math.max(max, prices[i] - min);
            min = Math.min(min, prices[i]);
        }
        return max;
    }

    //136. 只出现一次的数字
    public int singleNumber(int[] nums) {
        for (int i = 1; i < nums.length; i++) {
            nums[0] ^= nums[i];
        }
        return nums[0];
    }

    //是众数说明count不会被减为0 也就是如果被减为0说明不是众数 数学题
    public int majorityElement(int[] nums) {
        int count = 0, res = 0;
        for (int num : nums) {
            if (count == 0) {
                res = num;
            }
            if (num == res) {
                count++;
            } else count--;
        }
        return res;
    }

    //283. 移动零
    public void moveZeroes(int[] nums) {
        int zero = 0, noZero = 0;
        while (noZero < nums.length) {
            while (zero < nums.length && nums[zero] != 0) zero++;
            while (noZero < nums.length && nums[noZero] == 0) noZero++;
            if (zero >= nums.length || noZero >= nums.length) break;
            if (zero < noZero) {
                nums[zero] = nums[zero] ^ nums[noZero];
                nums[noZero] = nums[zero] ^ nums[noZero];
                nums[zero] = nums[zero] ^ nums[noZero];
            } else {
//                zero++;
                noZero++;
            }
        }
        SortTest listUtile = new SortTest();
        listUtile.dis(nums);
    }

    public List<List<Integer>> permute(int[] nums) {
        int[] vi = new int[nums.length];
        List<List<Integer>> allres = new ArrayList<>();
        List<Integer> res = new ArrayList<>();

        allNums2(nums, 0, res, allres);
        for (List<Integer> allre : allres) {
            System.out.println(allre);
        }
        return allres;
    }

    //31. 下一个排列
    public void nextPermutation(int[] nums) {
        int[] vi = new int[nums.length];
        List<Integer> res = new ArrayList<>();
//        allNums(nums, vi, res);

    }

    public void allNums(int[] nums, int[] vi, List<Integer> res, List<List<Integer>> allres) {
        if (res.size() == nums.length) {
            allres.add(new ArrayList<>(res));
            System.out.println(res);
        }
        for (int j = 0; j < nums.length; j++) {
            if (vi[j] == 1) continue;
            res.add(nums[j]);
            vi[j] = 1;
            allNums(nums, vi, res, allres);
            res.remove(res.size() - 1);
            vi[j] = 0;
        }
    }

    public void allNums2(int[] nums, int i, List<Integer> res, List<List<Integer>> allres) {
        if (i == nums.length - 1) {
            allres.add(Arrays.stream(nums).boxed().collect(Collectors.toList()));
//            System.out.println(res);
        }
        for (int j = i; j < nums.length; j++) {
//            res.add(nums[j]);
//            swap(nums[i], nums[j]);
            int t = nums[i];
            nums[i] = nums[j];
            nums[j] = t;

//            nums[i] = nums[i] ^ nums[j];
//            nums[j] = nums[i] ^ nums[j];
//            nums[i] = nums[i] ^ nums[j];
            allNums2(nums, i + 1, res, allres);
//            swap(nums[i], nums[j]);
//            nums[i] = nums[i] ^ nums[j];
//            nums[j] = nums[i] ^ nums[j];
//            nums[i] = nums[i] ^ nums[j];
            t = nums[i];
            nums[i] = nums[j];
            nums[j] = t;

        }
    }

    //    34. 在排序数组中查找元素的第一个和最后一个位置
    //    二分查找 lowerBound upperBound实现
    public int[] searchRange(int[] nums, int target) {
        int lower = lowerBound(nums, target);
        int upper = upperBound(nums, target) - 1;
        if (lower > upper || lower < 0 || lower >= nums.length || upper >= nums.length) {
            return new int[]{-1, -1};
        }
        System.out.println(lower + "，" + upper);
        return new int[]{lower, upper};
    }

    public int lowerBound(int[] nums, int target) {
        int l = 0, r = nums.length;
        while (l < r) {//跳出条件是l==r 也就是[nums.length,nums.length)
            int mid = (l + r) / 2;
            if (target == nums[mid]) {
                r = mid;// 右边是闭区间 此时变成[l,mid)
            } else if (target < nums[mid]) {
                r = mid;
            } else if (target > nums[mid]) {
                l = mid + 1;
            }
        }
        return l;//正常的话需要加判断nums[l]!= target? -1 : l;

    }

    public int upperBound(int[] nums, int target) {
        int l = 0, r = nums.length;
        while (l < r) {//跳出条件是l==r 也就是[nums.length,nums.length)
            int mid = (l + r) / 2;
            if (target == nums[mid]) {
                l = mid + 1;// 右边是闭区间 此时变成[l,mid)
            } else if (target < nums[mid]) {
                r = mid;
            } else if (target > nums[mid]) {
                l = mid + 1;
            }
        }
        return l;//正常的话需要加判断nums[l-1]!= target? -1 : l;
    }

    //704. 二分查找
    public int search(int[] nums, int target) {

//        实际上就是lowerBound

        int l = 0, r = nums.length, mid;
        while (l < r) {
            mid = (l + r) / 2;
            System.out.println(l + "," + r + "  mid:" + mid);

            if (nums[mid] == target) {
                r = mid;
            } else if (nums[mid] < target) {
                l = mid + 1;
            } else if (nums[mid] > target) {
                r = mid;
            }
        }
        return l >= nums.length || nums[l] != target ? -1 : l;
    }


    //50. Pow(x, n)快速幂
    public double myPow(double x, int n) {
        long N = n;
        return N < 0 ? 1.0 / quickMul(x, -N) : quickMul(x, N);
    }

    public double quickMul(double x, long N) {
        double res = 1.0;
        double tmp = x;

        while (N > 0) {
            if (N % 2 == 1) {
                res *= tmp;
            }
            tmp *= tmp;
            N /= 2;
        }
        return res;
    }

    //    347. 前 K 个高频元素 堆的使用（优先队列）
    public int[] topKFrequent(int[] nums, int k) {
        Map<Integer, Integer> map = new HashMap<Integer, Integer>();
        for (int num : nums) {
            map.put(num, map.getOrDefault(num, 0) + 1);
        }

        PriorityQueue<Pair<Integer, Integer>> heap = new PriorityQueue<>(Comparator.comparingInt(Pair::getValue));
        map.forEach((key, val) -> {
            if (heap.size() < k) {
                heap.add(new Pair<>(key, val));
            } else if (heap.peek().getValue() < val) {
                heap.poll();
                heap.add(new Pair<>(key, val));
            }
        });

        List<Integer> res = new ArrayList<>();
        heap.forEach(p -> res.add(p.getKey()));
        System.out.println(res);
        return res.stream().mapToInt(Integer::intValue).toArray();
    }

//75. 颜色分类 双指针

    public void sortColors(int[] nums) {
        int l = 0, r = nums.length - 1;
        while (l < nums.length - 1 && nums[l] == 0) l++;
        while (r > 0 && nums[r] == 2) r--;//坑

        for (int i = l; i <= r; ) {//坑
            int t;
            if (nums[i] == 0 && i > l) {
//                t = nums[i];
//                nums[i] = nums[l];
//                nums[l] = t;
                nums[i] = nums[i] ^ nums[l];
                nums[l] = nums[i] ^ nums[l];
                nums[i] = nums[i] ^ nums[l];
                while (l < nums.length - 1 && nums[l] == 0) l++;
            } else if (nums[i] == 2 && i < r) {
                nums[i] = nums[i] ^ nums[r];
                nums[r] = nums[i] ^ nums[r];
                nums[i] = nums[i] ^ nums[r];
//                t = nums[i];
//                nums[i] = nums[r];
//                nums[r] = t;
                while (r > 1 && nums[r] == 2) r--;
            } else i++;//坑
        }
        new SortTest().dis(nums);
    }


//    找出并返回这两个正序数组的 中位数

    public void findMid(int[] a, int[] b) {
        int mid1 = (a.length + b.length) >> 1;
        int mid2 = (a.length + b.length) % 2 == 0 ? mid1 - 1 : 0;
        Stack<Integer> stack = new Stack<>();
        int count = 0, i = 0, j = 0;
        while (count <= mid1 && i < a.length && j < b.length) {
            stack.push(a[i] <= b[j] ? a[i++] : b[j++]);
            count++;
        }
        while (count <= mid1 && i < a.length) {
            stack.push(a[i++]);
            count++;
        }

        while (count <= mid1 && j < b.length) {
            stack.push(b[j++]);
            count++;
        }
        double res = (a.length + b.length) % 2 != 0 ? stack.pop() : (stack.pop() + stack.pop()) / 2.0;
        System.out.println(res);

    }

    //    public static List<Integer> findPrime() {
//        List<Integer> list = new ArrayList<>(100000);
//        for (int n = 2; n < 1000000; n++) {
//            boolean isPrime = true;
//            int sqrt = (int) Math.sqrt(n);
//            for (Integer i : list) {
//                if (n % i == 0) {
//                    isPrime = false;
//                    break;
//                }
//                if (i > sqrt) {
//                    break;
//                }
//            }
//            if (isPrime) {
//                list.add(n);
//            }
//        }
//    查找范围内的素数 使用素数定理：一个素数不能被小于自己的素数整除
    public static List<Integer> findPrime() {
        List<Integer> res = new ArrayList<>();
        for (int i = 2; i < 100; i++) {
            int tmpi = i;
            boolean is = Optional.ofNullable(res)
                    .filter(r -> r.size() > 0)
                    .map(r -> r.stream().noneMatch(val -> tmpi % val == 0))
                    .orElse(true);
            if (is) {
                res.add(i);
            }
        }
        System.out.println(res);
        return res;
    }

    //剑指 Offer 57 - II. 和为s的连续正
    public int[][] findContinuousSequence(int target) {
        List<int[]> res = new ArrayList<>();
        int l = 1, r = 1;
        List<Integer> list = IntStream.range(0, target).boxed().collect(Collectors.toList());
        while (l < target && r < target && l <= r) {
            int sum = list.subList(l, r + 1).stream().mapToInt(Integer::intValue).sum();
            if (sum == target) {
                res.add(list.subList(l, r + 1).stream().mapToInt(Integer::intValue).toArray());
                System.out.println(l + "..." + r);
                r++;
            } else if (sum < target) {
                r++;
            } else if (sum > target) {
                l++;
            }
        }
//        MyUtile.disMap(res);
        return res.toArray(new int[res.size()][]);

    }

    //    154. 寻找旋转排序数组中的最小值 II
    public int findMin(int[] nums) {
        int l = 0, r = nums.length - 1;
        while (l < r) {
            int mid = l + (r - l) / 2;
            if (nums[mid] < nums[r]) {
                r = mid;
            } else if (nums[mid] > nums[r]) {
                l = mid + 1;
            } else {
                r--;
            }
        }
        return nums[l];

    }

    public String replaceSpace(String s) {
        return s.replaceAll(" ", "%20");

    }

    public int[] exchange(int[] nums) {
        int j = 0, o = nums.length - 1;

        while (j < nums.length && o < nums.length && j < o) {

            while (j < nums.length && nums[j] % 2 == 1) {
                j++;
            }
            while (o >= 0 && nums[o] % 2 == 0) {
                o--;
            }
            if (j < nums.length && o < nums.length && j < o) {
                nums[j] = nums[j] ^ nums[o];
                nums[o] = nums[j] ^ nums[o];
                nums[j] = nums[j] ^ nums[o];
            }
        }
        MyUtile.dis(nums);
        return nums;

    }


    //    剑指 Offer 17. 打印从1到最大的n位数 全排列回溯
    public int[] printNumbers(int n) {


        List<String> res = new ArrayList<>();
        int[] list = new int[]{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
        for (int i = 1; i <= n; i++) {
            dfs(0, i, new String(), list, res);
        }

        System.out.println(res);
        return res.stream().mapToInt(Integer::parseInt).toArray();
//        return res.stream().mapToInt(String::int).toArray();

    }

    public void dfs(int x, int len, String tmplist, int[] list, List<String> res) {
        if (x == len) {
            res.add(tmplist.toString());
            return;
        }
        int start = x == 0 ? 1 : 0; // X=0表示全排列左边第一位数字，不能为0 即使是一位数的时候也是不能为0的
        for (int i = start; i < 10; i++) {
            tmplist += list[i];
            dfs(x + 1, len, tmplist, list, res);
            tmplist = tmplist.substring(0, tmplist.length() - 1);
        }

    }

    //    剑指 Offer 40. 最小的k个数 堆使用
    public int[] getLeastNumbers(int[] arr, int k) {
        PriorityQueue<Integer> heap = new PriorityQueue<Integer>((a, b) -> b - a);
        for (int i = 0; i < arr.length; i++) {
            if (heap.size() >= k) {
                if (!heap.isEmpty() && arr[i] < heap.peek()) {
                    heap.add(arr[i]);
                    heap.poll();
                }
            } else {
                heap.add(arr[i]);
            }
        }
        System.out.println(heap);
        return heap.stream().mapToInt(Integer::intValue).toArray();

    }

    //    剑指 Offer 65. 不用加减乘除做加法
    public int add(int a, int b) {
        int carry;
        while (b != 0) {
            carry = (a & b) << 1;
            a ^= b;
            b = carry;
        }
        System.out.println(a);
        return a;

    }
//    剑指 Offer 62. 圆圈中最后剩下的数字

    public int lastRemaining(int n, int m) {

        List<Integer> list = IntStream.range(0, n).boxed().collect(Collectors.toList());
        int d = 0;
        while (list.size() > 1) {
            int del = (d + m - 1) % list.size();
            System.out.println(list.get(del));
            list.remove(del);
            d = (del) % list.size();
        }
        System.out.println(list);
        return list.get(0);
    }

    //    剑指 Offer II 003. 前 n 个数字二进制中 1 的个数
    public int[] countBits(int n) {
        ArrayList<Integer> list = new ArrayList<>();
        Stream.iterate(0, i -> i + 1).limit(n + 1)
                .forEach(i -> list.add(countOne(i)));
        System.out.println(list);
        return list.stream().mapToInt(Integer::intValue).toArray();
    }

    //计算二进制1的个数高效方法
    public int countOne(int n) {
        int count = 0;
        while (n > 0) {
            n &= n - 1;
            count++;
        }
        return count;
    }

    //    剑指 Offer II 068. 查找插入位置
    public int searchInsert(int[] nums, int target) {
        return lowerbound(nums, 0, nums.length, target);

    }

    public int lowerbound(int[] nums, int l, int r, int target) {
        while (l < r) {
//            [)
            int mid = l + (r - l) / 2;
            if (nums[mid] == target) {
                r = mid;
            } else if (target < nums[mid]) {
                r = mid;
            } else if (target > nums[mid]) {
                l = mid + 1;
            }
        }
        return l;
    }

    //    剑指 Offer 04. 二维数组中的查找
    public boolean findNumberIn2DArray(int[][] matrix, int target) {
        if (matrix.length == 0) {
            return false;
        }
        int i = 0, j = matrix[0].length - 1;
        while (i < matrix.length && j >= 0) {
            if (matrix[i][j] == target) {
                System.out.println(i + "," + j);
                return true;
            } else if (matrix[i][j] < target) {
                i++;
            } else if (matrix[i][j] > target) {
                j--;
            }
        }
        return false;

    }


    //4. 寻找两个正序数组的中位数 二分 等同于找第k和k+1位数
    public double findMedianSortedArrays(int[] nums1, int[] nums2) {
        int length1 = nums1.length, length2 = nums2.length;
        int totalLength = length1 + length2;
        if (totalLength % 2 == 1) {
            int midIndex = totalLength / 2;
            double median = getKthElement(nums1, nums2, midIndex + 1);
            return median;
        } else {
            int midIndex1 = totalLength / 2 - 1, midIndex2 = totalLength / 2;
            double median = (getKthElement(nums1, nums2, midIndex1 + 1) + getKthElement(nums1, nums2, midIndex2 + 1)) / 2.0;
            return median;
        }
    }

    public int getKthElement(int[] nums1, int[] nums2, int k) {
        /* 主要思路：要找到第 k (k>1) 小的元素，那么就取 pivot1 = nums1[k/2-1] 和 pivot2 = nums2[k/2-1] 进行比较
         * 这里的 "/" 表示整除
         * nums1 中小于等于 pivot1 的元素有 nums1[0 .. k/2-2] 共计 k/2-1 个
         * nums2 中小于等于 pivot2 的元素有 nums2[0 .. k/2-2] 共计 k/2-1 个
         * 取 pivot = min(pivot1, pivot2)，两个数组中小于等于 pivot 的元素共计不会超过 (k/2-1) + (k/2-1) <= k-2 个
         * 这样 pivot 本身最大也只能是第 k-1 小的元素
         * 如果 pivot = pivot1，那么 nums1[0 .. k/2-1] 都不可能是第 k 小的元素。把这些元素全部 "删除"，剩下的作为新的 nums1 数组
         * 如果 pivot = pivot2，那么 nums2[0 .. k/2-1] 都不可能是第 k 小的元素。把这些元素全部 "删除"，剩下的作为新的 nums2 数组
         * 由于我们 "删除" 了一些元素（这些元素都比第 k 小的元素要小），因此需要修改 k 的值，减去删除的数的个数
         */

        int length1 = nums1.length, length2 = nums2.length;
        int index1 = 0, index2 = 0;
        int kthElement = 0;

        while (true) {
            // 边界情况
            if (index1 == length1) {
                return nums2[index2 + k - 1];
            }
            if (index2 == length2) {
                return nums1[index1 + k - 1];
            }
            if (k == 1) {
                return Math.min(nums1[index1], nums2[index2]);
            }

            // 正常情况
            int half = k / 2;
            int newIndex1 = Math.min(index1 + half, length1) - 1;
            int newIndex2 = Math.min(index2 + half, length2) - 1;
            int pivot1 = nums1[newIndex1], pivot2 = nums2[newIndex2];
            if (pivot1 <= pivot2) {
                k -= (newIndex1 - index1 + 1);
                index1 = newIndex1 + 1;
            } else {
                k -= (newIndex2 - index2 + 1);
                index2 = newIndex2 + 1;
            }
        }
    }


    //    4. 寻找两个正序数组的中位数
    public double findMedianSortedArrays2(int[] nums1, int[] nums2) {
        int totalLen = nums1.length + nums2.length;
        if (totalLen % 2 == 1) {
            return getKthElement2(nums1, nums2, totalLen / 2+1);//加一是因为k是第几个 没有包括0下标的
        } else {
            return (getKthElement2(nums1, nums2, totalLen / 2) + getKthElement2(nums1, nums2, totalLen / 2 + 1)) / 2.0;
        }
    }

    public int getKthElement2(int[] nums1, int[] nums2, int k) {
        int ind1 = 0, ind2 = 0;
        while (true) {
            if (ind1 == nums1.length) {
                return nums2[ind2 + (k - 1)];//减一是为了避免两边的前面个数加起来超过k，因为k是第几个 没有包括0下标的
            }
            if (ind2 == nums2.length) {
                return nums1[ind1 + (k - 1)];//减一是为了避免两边的前面个数加起来超过k，因为k是第几个 没有包括0下标的
            }
            if (k == 1) {
                return Math.min(nums1[ind1], nums2[ind2]);
            }
            int half = k / 2;
            int tempInd1 = Math.min(ind1 + half, nums1.length) - 1;//减一是为了避免两边的前面个数加起来超过k
            int tempInd2 = Math.min(ind2 + half, nums2.length) - 1;//减一是为了避免两边的前面个数加起来超过k
            if (nums1[tempInd1] <= nums2[tempInd2]) {
                k -= (tempInd1 - ind1 + 1);
                ind1 = tempInd1 + 1;
            } else {
                k -= (tempInd2 - ind2 + 1);
                ind2 = tempInd2 + 1;
            }
        }


    }


    public static void main(String[] args) {
        Numbers numbers = new Numbers();
//        System.out.println(numbers.mySqrt(0));
//        System.out.println(numbers.mySqrt(1));
//        System.out.println(numbers.mySqrt(2));
//        System.out.println(numbers.mySqrt(3));
//        System.out.println(numbers.mySqrt(4));
//        System.out.println(numbers.mySqrt(2147395599));
//        System.out.println(numbers.maxProfit(new int[]{7, 6, 4, 3, 1}));
//        System.out.println(numbers.majorityElement(new int[]{2, 2, 1, 1, 1, 2, 2}));
//        numbers.moveZeroes(new int[]{1, 0, 1});
//        numbers.searchRange(new int[]{5, 7, 7, 8, 8, 10}, 8);
//        System.out.println(numbers.s
//        numbers.topKFrequent(new int[]{1, 1, 1, 2, 2, 3}, 2);
//        numbers.sortColors(new int[]{2, 0, 1});
//        numbers.findMid(new int[]{1, 3}, new int[]{2, 4});
        int[] nums1 = new int[]{1, 3};
        int[] nums2 = new int[]{2};
        System.out.println(numbers.findMedianSortedArrays2(nums1, nums2));
    }

}


