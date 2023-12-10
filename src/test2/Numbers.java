package test2;

import category.MyUtile;
import javafx.util.Pair;

import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class Numbers {
    public static void main(String[] args) {
        Numbers numbers = new Numbers();
//        System.out.println(numbers.mySqrt(1));
        int[] nums = new int[]{3, 4, -1, 1};
        int[] nums2 = new int[]{2, 3, 1};
//        System.out.println(numbers.searchRange(new int[]{1}, 1)[0] + "," + numbers.searchRange(new int[]{1}, 1)[1]);
//        System.out.println(numbers.findDisappearedNumbers(nums));
//        numbers.nextPermutation(nums);
//        numbers.lastRemaining(5, 3);
//        numbers.setZeroes(new int[][]{{0,1,2,0},{3,4,5,2},{1,3,1,5}});
//        numbers.setZeroes(new int[][]{{1, 0}});
//        numbers.findContinuousSequence(15);
        System.out.println(numbers.firstMissingPositive(nums));

    }

    public static void swap(int[] nums, int a, int b) {
        int t = nums[a];
        nums[a] = nums[b];
        nums[b] = t;
    }

    //    69. x 的平方根 二分查找变形
    public int mySqrt(int x) {

        int l = 0, r = x, res = 0;
        while (l <= r) {
            int mid = l + (r - l) / 2;
            if ((long) mid * mid <= x) {
                res = mid;
                l = mid + 1;
            } else {
                r = mid - 1;
            }
        }
        return res;

    }

    //    34. 在排序数组中查找元素的第一个和最后一个位置
    public int[] searchRange(int[] nums, int target) {
        return new int[]{lowerbound(nums, 0, nums.length, target), upperbound(nums, 0, nums.length, target) - 1};
    }

    //    lowerbound
    public int lowerbound(int[] nums, int l, int r, int target) {
        while (l < r) {
//            [)
            int mid = l + (r - l) / 2;
            if (nums[mid] == target) {
                r = mid;
            } else if (nums[mid] < target) {
                l = mid + 1;
            } else if (nums[mid] > target) {
                r = mid;
            }
        }
        return nums.length > 0 && l < nums.length && nums[l] == target ? l : -1;
    }

    public int upperbound(int[] nums, int l, int r, int target) {
        while (l < r) {
//            [)
            int mid = l + (r - l) / 2;
            if (nums[mid] == target) {
                l = mid + 1;
            } else if (nums[mid] < target) {
                l = mid + 1;
            } else if (nums[mid] > target) {
                r = mid;
            }
        }
        return nums.length > 0 && l > 0 && l <= nums.length && nums[l - 1] == target ? l : 0;

    }

    //    //121. 买卖股票的最佳时机
    public int maxProfit(int[] prices) {
        int min = 99999, maxres = 0;
        for (int i = 0; i < prices.length; i++) {
            min = Math.min(min, prices[i]);
            maxres = Math.max(maxres, prices[i] - min);
        }
        return maxres;
    }

    //    122. 买卖股票的最佳时机 II
    public int maxProfit2(int[] prices) {
        int n = prices.length;
        int dpi0 = 0, dpi1 = -999999;
        for (int i = 0; i < n; i++) {
            int t = dpi0;
            dpi0 = Math.max(dpi0, dpi1 + prices[i]);
            dpi1 = Math.max(dpi1, t - prices[i]);
        }
        return dpi0;
    }

    //    123. 买卖股票的最佳时机 III
    public int maxProfit3(int[] prices) {
        int n = prices.length, i, j;
        int[][][] dp = new int[n][3][2];
//        dp[-1][][0]=0;
//        dp[-1][][1]=-99999;
//        dp[][0][0]=0;
//        dp[][0][1]=-99999;
        for (i = 0; i < n; i++) {
            for (j = 0; j < 3; j++) {
                if (j == 0) {
                    dp[i][j][0] = 0;
                    dp[i][j][1] = -9999;
                    continue;
                }
                if (i == 0) {
                    dp[i][j][0] = 0;
                    dp[i][j][1] = -prices[i];
                    continue;
                }

                dp[i][j][0] = Math.max(dp[i - 1][j][0], dp[i - 1][j][1] + prices[i]);
                dp[i][j][1] = Math.max(dp[i - 1][j][1], dp[i - 1][j - 1][0] - prices[i]);

            }

        }


        return dp[n - 1][2][0];
    }

    //188. 买卖股票的最佳时机 IV
    public int maxProfit(int k, int[] prices) {
        int n = prices.length;
        int[][][] dp = new int[n][k + 1][2];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < k + 1; j++) {
                if (i == 0) {
                    dp[0][j][0] = 0;
                    dp[0][j][1] = -prices[i];
                    continue;
                }
                if (j == 0) {
                    dp[i][0][0] = 0;
                    dp[i][0][1] = -99999;
                    continue;
                }
                dp[i][j][0] = Math.max(dp[i - 1][j][0], dp[i - 1][j][1] + prices[i]);
                dp[i][j][1] = Math.max(dp[i - 1][j][1], dp[i - 1][j - 1][0] - prices[i]);
            }
        }
        return dp[n - 1][k][0];

    }

//309. 最佳买卖股票时机含冷冻期

    public int maxProfit4(int[] prices) {

        int n = prices.length, k = 1;
        int[][][] dp = new int[n][k + 1][2];
        for (int i = 0; i < n; i++) {
            for (int j = 1; j < k + 1; j++) {
                if (i == 0) {
                    dp[0][j][0] = 0;
                    dp[0][j][1] = -prices[i];
                    continue;
                }
                if (i == 1) {
                    dp[1][j][0] = Math.max(dp[i - 1][j][0], dp[i - 1][j][1] + prices[i]);
                    dp[1][j][1] = Math.max(dp[i - 1][j][1], -prices[i]);
                    continue;
                }
//                if (j == 0) {
//                    dp[i][0][0] = 0;
//                    dp[i][0][1] = -99999;
//                    continue;
//                }
                dp[i][j][0] = Math.max(dp[i - 1][j][0], dp[i - 1][j][1] + prices[i]);
                dp[i][j][1] = Math.max(dp[i - 1][j][1], dp[i - 2][j][0] - prices[i]);
            }
        }
        System.out.println(dp[n - 1][k][0]);
        return dp[n - 1][k][0];


    }

    //    714. 买卖股票的最佳时机含手续费
    public int maxProfit5(int[] prices, int fee) {

        int n = prices.length, k = 1;
        int[][][] dp = new int[n][k + 1][2];
        for (int i = 0; i < n; i++) {
            for (int j = 1; j < k + 1; j++) {
                if (i == 0) {
                    dp[0][j][0] = 0;
                    dp[0][j][1] = -prices[i];
                    continue;
                }
                dp[i][j][0] = Math.max(dp[i - 1][j][0], dp[i - 1][j][1] + prices[i] - fee);
                dp[i][j][1] = Math.max(dp[i - 1][j][1], dp[i - 1][j][0] - prices[i]);
            }
        }
        System.out.println(dp[n - 1][k][0]);
        return dp[n - 1][k][0];
    }

    //136. 只出现一次的数字
    public int singleNumber(int[] nums) {
        int res = 0;
        for (int i : nums) {
            res ^= i;
        }
        return res;

    }

    //    //是众数说明count不会被减为0 也就是如果被减为0说明不是众数 数学题
    public int majorityElement(int[] nums) {
        int count = 0, res = 0;
        for (int num : nums) {
            if (count == 0) {
                res = num;
            }
            count += res == num ? 1 : -1;
        }
        return res;
    }

    //    //283. 移动零
    public void moveZeroes(int[] nums) {
        int n = nums.length;

        int zero = 0, nozero = 0;
        while (zero < n && nums[zero] != 0) {
            zero++;
        }
        nozero = zero;
        while (nozero < n && nums[nozero] == 0) {
            nozero++;
        }
        while (zero < n && nozero < n) {
            swap(nums, zero, nozero);
            while (zero < n && nums[zero] != 0) {
                zero++;
            }
            nozero = zero;
            while (nozero < n && nums[nozero] == 0) {
                nozero++;
            }
        }
//        MyUtile.dis(nums);

    }

    //
    public List<List<Integer>> permute(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        allNums(nums, 0, res);
        System.out.println(res);
        return res;
    }

    //    //31. 下一个排列 根据数字组合的特征解决
//    // 1.如果序列全部降序说明这个序列已经是最大的了
//    // 2.从后往前如果升序到某一个地方突然某一个数k降下去说明这个序列不是最大的还可以提升 并且使用k与后面的序列中的一个大于k的最小值交换然后后面重排为升序就是下一个序列
    public void nextPermutation(int[] nums) {
        int n = nums.length, i;
        List<Integer> res = Arrays.stream(nums).boxed().collect(Collectors.toList());
        for (i = n - 1; i > 0 && nums[i] <= nums[i - 1]; i--) ;
        if (i == 0) {
            MyUtile.reverse(nums, 0, n - 1);
        } else {
            int k = i - 1, j;
            for (j = i; j < n && nums[j] > nums[k]; j++) ;
            swap(nums, k, j - 1);
            MyUtile.reverse(nums, k + 1, n - 1);
        }
        MyUtile.dis(nums);
    }


    //
//    //    全排列
    public void allNums(int[] nums, int i, List<List<Integer>> res) {
        if (i == nums.length) {
            res.add(Arrays.stream(nums).boxed().collect(Collectors.toList()));
        }
        for (int j = i; j < nums.length; j++) {
            swap(nums, i, j);
            allNums(nums, i + 1, res);
            swap(nums, i, j);
        }
    }


    //    //704. 二分查找
    public int search(int[] nums, int target) {
        int l = 0, r = nums.length;
        while (l < r) {
            int mid = l + (r - l) / 2;
            if (nums[mid] == target) {
                r = mid;
            } else if (nums[mid] < target) {
                l = mid + 1;
            } else if (nums[mid] > target) {
                r = mid;
            }
        }


        return nums.length > 0 && l >= 0 && l < nums.length && nums[l] == target ? l : -1;

    }


    //    //50. Pow(x, n)快速幂
    public double myPow(double x, int n) {
        return n >= 0 ? quickMul(x, (long) n) : 1.0 / quickMul(x, -(long) n);
    }

    public double quickMul(double x, long n) {
        double res = 0, base = x;
        while (n != 0) {
            if (n % 2 == 1) {
                res += base;
            }
            base *= base;
            n /= 2;
        }
        return res;
    }

    //    //    347. 前 K 个高频元素 堆的使用（优先队列）
    public int[] topKFrequent(int[] nums, int k) {
        PriorityQueue<Pair<Integer, Integer>> heap = new PriorityQueue<>((o1, o2) -> o1.getValue() - o2.getValue());
        Map<Integer, Integer> map = new HashMap<>();

        for (int num : nums) {
            map.put(num, map.getOrDefault(num, 0) + 1);
        }
        map.forEach((ke, v) -> {
            heap.add(new Pair<>(ke, v));
            if (heap.size() > k && Objects.nonNull(heap.poll())) ;
        });
        return heap.stream().mapToInt(Pair::getKey).toArray();
    }

    ////75. 颜色分类 双指针
//
    public void sortColors(int[] nums) {
        int l = 0, n = nums.length, r = n - 1;
        for (int i = 0; i <= r; ) {
            if (nums[i] == 2) {
                while (i < r && nums[r] == 2) {
                    r--;
                }
                swap(nums, i, r);

                r--;
            } else if (nums[i] == 0 && i != l) {
                swap(nums, i, l);
                l++;
            } else {
                i++;
            }
        }
        MyUtile.dis(nums);

    }

    //
//
////    找出并返回这两个正序数组的 中位数
//
//    public void findMid(int[] a, int[] b) {
//        int mid1 = (a.length + b.length) >> 1;
//        int mid2 = (a.length + b.length) % 2 == 0 ? mid1 - 1 : 0;
//        Stack<Integer> stack = new Stack<>();
//        int count = 0, i = 0, j = 0;
//        while (count <= mid1 && i < a.length && j < b.length) {
//            stack.push(a[i] <= b[j] ? a[i++] : b[j++]);
//            count++;
//        }
//        while (count <= mid1 && i < a.length) {
//            stack.push(a[i++]);
//            count++;
//        }
//
//        while (count <= mid1 && j < b.length) {
//            stack.push(b[j++]);
//            count++;
//        }
//        double res = (a.length + b.length) % 2 != 0 ? stack.pop() : (stack.pop() + stack.pop()) / 2.0;
//        System.out.println(res);
//
//    }
//
//    //    public static List<Integer> findPrime() {
////        List<Integer> list = new ArrayList<>(100000);
////        for (int n = 2; n < 1000000; n++) {
////            boolean isPrime = true;
////            int sqrt = (int) Math.sqrt(n);
////            for (Integer i : list) {
////                if (n % i == 0) {
////                    isPrime = false;
////                    break;
////                }
////                if (i > sqrt) {
////                    break;
////                }
////            }
////            if (isPrime) {
////                list.add(n);
////            }
////        }
////    查找范围内的素数 使用素数定理：一个素数不能被小于自己的素数整除
//    public static List<Integer> findPrime() {
//        List<Integer> res = new ArrayList<>();
//        for (int i = 2; i < 100; i++) {
//            int tmpi = i;
//            boolean is = Optional.ofNullable(res)
//                    .filter(r -> r.size() > 0)
//                    .map(r -> r.stream().noneMatch(val -> tmpi % val == 0))
//                    .orElse(true);
//            if (is) {
//                res.add(i);
//            }
//        }
//        System.out.println(res);
//        return res;
//    }
//
    //剑指 Offer 57 - II. 和为s的连续正
    public int[][] findContinuousSequence(int target) {
//        int[][] res=new int[target][];
        List<int[]> res = new ArrayList<>();
        int l = 1, r = 1, tmp = 0;
        while (l < target && r < target) {
            tmp = Stream.iterate(l, v -> v + 1).limit(r - l + 1).mapToInt(Integer::intValue).sum();
            if (tmp == target) {
                res.add(Stream.iterate(l, v -> v + 1).limit(r - l + 1).mapToInt(Integer::intValue).toArray());
                r++;
            } else if (tmp < target) {
                r++;
            } else if (tmp > target) {
                l++;
            }
        }
//        for (int i = 0; i < res.size(); i++) {
//            MyUtile.dis(res.get(i));
//        }
//        System.out.println(res);
        return res.toArray(new int[0][]);
    }

    //
//    //    154. 寻找旋转排序数组中的最小值 II
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

    //
//    public String replaceSpace(String s) {
//        return s.replaceAll(" ", "%20");
//
//    }
//
//    public int[] exchange(int[] nums) {
//        int j = 0, o = nums.length - 1;
//
//        while (j < nums.length && o < nums.length && j < o) {
//
//            while (j < nums.length && nums[j] % 2 == 1) {
//                j++;
//            }
//            while (o >= 0 && nums[o] % 2 == 0) {
//                o--;
//            }
//            if (j < nums.length && o < nums.length && j < o) {
//                nums[j] = nums[j] ^ nums[o];
//                nums[o] = nums[j] ^ nums[o];
//                nums[j] = nums[j] ^ nums[o];
//            }
//        }
//        MyUtile.dis(nums);
//        return nums;
//
//    }
//
//
//    //    剑指 Offer 17. 打印从1到最大的n位数 全排列回溯
//    public int[] printNumbers(int n) {
//
//
//        List<String> res = new ArrayList<>();
//        int[] list = new int[]{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
//        for (int i = 1; i <= n; i++) {
//            dfs(0, i, new String(), list, res);
//        }
//
//        System.out.println(res);
//        return res.stream().mapToInt(Integer::parseInt).toArray();
////        return res.stream().mapToInt(String::int).toArray();
//
//    }
//
//    public void dfs(int x, int len, String tmplist, int[] list, List<String> res) {
//        if (x == len) {
//            res.add(tmplist.toString());
//            return;
//        }
//        int start = x == 0 ? 1 : 0; // X=0表示全排列左边第一位数字，不能为0 即使是一位数的时候也是不能为0的
//        for (int i = start; i < 10; i++) {
//            tmplist += list[i];
//            dfs(x + 1, len, tmplist, list, res);
//            tmplist = tmplist.substring(0, tmplist.length() - 1);
//        }
//
//    }
//
//    //    剑指 Offer 40. 最小的k个数 堆使用
    public int[] getLeastNumbers(int[] arr, int k) {
        PriorityQueue<Integer> heap = new PriorityQueue<>((o1, o2) -> o2 - o1);
        if (k == 0) {
            return new int[]{};
        }
        for (int num : arr) {
            if (heap.size() < k) {
                heap.offer(num);
            } else {
                if (heap.peek() > num) {
                    heap.poll();
                    heap.offer(num);
                }
            }
        }
        return new ArrayList<>(heap).stream().mapToInt(Integer::intValue).toArray();

    }

    //    //    剑指 Offer 65. 不用加减乘除做加法
    public int add(int a, int b) {
        int c = 0;
        while (b != 0) {
            c = (a & b) << 1;
            a ^= b;
            b = c;
        }
        return a;


    }

    ////    剑指 Offer 62. 圆圈中最后剩下的数字
//
    public int lastRemaining(int n, int m) {
        List<Integer> list = Stream.iterate(0, t -> t + 1).limit(n).collect(Collectors.toList());
        int k = 0;
        while (list.size() > 1) {
            int del = (k + m - 1) % list.size();
            k = del;
            System.out.println(list.get(del));
            list.remove(del);
        }
        return list.get(0);

//        List<Integer> list = IntStream.range(0, n).boxed().collect(Collectors.toList());
//        int d = 0;
//        while (list.size() > 1) {
//            int del = (d + m - 1) % list.size();
//            System.out.println(list.get(del));
//            list.remove(del);
//            d = (del) % list.size();
//        }
//        System.out.println(list);
//        return list.get(0);
    }

    //
//    //    剑指 Offer II 003. 前 n 个数字二进制中 1 的个数  338. 比特位计数
    public int[] countBits(int n) {
        ArrayList<Integer> list = new ArrayList<>();
        Stream.iterate(0, i -> i + 1).limit(n + 1)
                .forEach(i -> list.add(countOne(i)));
        System.out.println(list);
        return list.stream().mapToInt(Integer::intValue).toArray();
    }

    //    //计算二进制1的个数高效方法
    public int countOne(int n) {
        int res = 0;
        while (n > 0) {
            n &= (n - 1);
            res++;
        }
        return res;
    }


    //    //    剑指 Offer 04. 二维数组中的查找
    public boolean findNumberIn2DArray(int[][] matrix, int target) {
        if (matrix.length == 0) {
            return false;
        }
        int i = 0, j = matrix[0].length - 1;
        while (i >= 0 && i < matrix.length && j >= 0 && j < matrix[0].length) {
            if (matrix[i][j] == target) {
                return true;
            } else if (matrix[i][j] < target) {
                i++;
            } else {
                j--;
            }
        }
        return false;
    }

    //
//
//    //4. 寻找两个正序数组的中位数 二分 等同于找第k和k+1位数
//    public double findMedianSortedArrays(int[] nums1, int[] nums2) {
//        int length1 = nums1.length, length2 = nums2.length;
//        int totalLength = length1 + length2;
//        if (totalLength % 2 == 1) {
//            int midIndex = totalLength / 2;
//            double median = getKthElement(nums1, nums2, midIndex + 1);
//            return median;
//        } else {
//            int midIndex1 = totalLength / 2 - 1, midIndex2 = totalLength / 2;
//            double median = (getKthElement(nums1, nums2, midIndex1 + 1) + getKthElement(nums1, nums2, midIndex2 + 1)) / 2.0;
//            return median;
//        }
//    }
//
//    public int getKthElement(int[] nums1, int[] nums2, int k) {
//        /* 主要思路：要找到第 k (k>1) 小的元素，那么就取 pivot1 = nums1[k/2-1] 和 pivot2 = nums2[k/2-1] 进行比较
//         * 这里的 "/" 表示整除
//         * nums1 中小于等于 pivot1 的元素有 nums1[0 .. k/2-2] 共计 k/2-1 个
//         * nums2 中小于等于 pivot2 的元素有 nums2[0 .. k/2-2] 共计 k/2-1 个
//         * 取 pivot = min(pivot1, pivot2)，两个数组中小于等于 pivot 的元素共计不会超过 (k/2-1) + (k/2-1) <= k-2 个
//         * 这样 pivot 本身最大也只能是第 k-1 小的元素
//         * 如果 pivot = pivot1，那么 nums1[0 .. k/2-1] 都不可能是第 k 小的元素。把这些元素全部 "删除"，剩下的作为新的 nums1 数组
//         * 如果 pivot = pivot2，那么 nums2[0 .. k/2-1] 都不可能是第 k 小的元素。把这些元素全部 "删除"，剩下的作为新的 nums2 数组
//         * 由于我们 "删除" 了一些元素（这些元素都比第 k 小的元素要小），因此需要修改 k 的值，减去删除的数的个数
//         */
//
//        int length1 = nums1.length, length2 = nums2.length;
//        int index1 = 0, index2 = 0;
//        int kthElement = 0;
//
//        while (true) {
//            // 边界情况
//            if (index1 == length1) {
//                return nums2[index2 + k - 1];
//            }
//            if (index2 == length2) {
//                return nums1[index1 + k - 1];
//            }
//            if (k == 1) {
//                return Math.min(nums1[index1], nums2[index2]);
//            }
//
//            // 正常情况
//            int half = k / 2;
//            int newIndex1 = Math.min(index1 + half, length1) - 1;
//            int newIndex2 = Math.min(index2 + half, length2) - 1;
//            int pivot1 = nums1[newIndex1], pivot2 = nums2[newIndex2];
//            if (pivot1 <= pivot2) {
//                k -= (newIndex1 - index1 + 1);
//                index1 = newIndex1 + 1;
//            } else {
//                k -= (newIndex2 - index2 + 1);
//                index2 = newIndex2 + 1;
//            }
//        }
//    }
//
//
//    //    4. 寻找两个正序数组的中位数
//    public double findMedianSortedArrays2(int[] nums1, int[] nums2) {
//        int totalLen = nums1.length + nums2.length;
//        if (totalLen % 2 == 1) {
//            return getKthElement2(nums1, nums2, totalLen / 2 + 1);//加一是因为k是第几个 没有包括0下标的
//        } else {
//            return (getKthElement2(nums1, nums2, totalLen / 2) + getKthElement2(nums1, nums2, totalLen / 2 + 1)) / 2.0;
//        }
//    }
//
//
//    public int getKthElement2(int[] nums1, int[] nums2, int k) {
//        int ind1 = 0, ind2 = 0;
//        while (true) {
//            if (ind1 == nums1.length) {
//                return nums2[ind2 + (k - 1)];//减一是为了避免两边的前面个数加起来超过k，因为k是第几个 没有包括0下标的
//            }
//            if (ind2 == nums2.length) {
//                return nums1[ind1 + (k - 1)];//减一是为了避免两边的前面个数加起来超过k，因为k是第几个 没有包括0下标的
//            }
//            if (k == 1) {
//                return Math.min(nums1[ind1], nums2[ind2]);
//            }
//            int half = k / 2;
//            int tempInd1 = Math.min(ind1 + half, nums1.length) - 1;//减一是为了避免两边的前面个数加起来超过k
//            int tempInd2 = Math.min(ind2 + half, nums2.length) - 1;//减一是为了避免两边的前面个数加起来超过k
//            if (nums1[tempInd1] <= nums2[tempInd2]) {
//                k -= (tempInd1 - ind1 + 1);
//                ind1 = tempInd1 + 1;
//            } else {
//                k -= (tempInd2 - ind2 + 1);
//                ind2 = tempInd2 + 1;
//            }
//        }
//
//
//    }
//
//    //448. 找到所有数组中消失的数字 利用下标
    public List<Integer> findDisappearedNumbers(int[] nums) {
        int n = nums.length;
        for (int i = 0; i < n; i++) {
            int ind = nums[i] > n ? nums[i] - n - 1 : nums[i] - 1;
            nums[ind] += nums[ind] > n ? 0 : n;
        }
        List<Integer> res = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            if (nums[i] <= n) res.add(i + 1);
        }

        return res;

    }

    //    //461. 汉明距离
    public int hammingDistance(int x, int y) {
        return countOne(x ^ y);
    }

    //    //48. 旋转图像  https://labuladong.github.io/algo/di-yi-zhan-da78c/shou-ba-sh-48c1d/er-wei-shu-150fb/
    public void rotate(int[][] matrix) {
        int m = matrix.length, n = matrix[0].length;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < i; j++) {
                int t = matrix[i][j];
                matrix[i][j] = matrix[j][i];
                matrix[j][i] = t;
            }
        }
        MyUtile.disMap(matrix);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n / 2; j++) {
                swap(matrix[i], j, n - j);
            }
        }

//        MyUtile.disMap(matrix);
    }

    //    //54. 螺旋矩阵  https://labuladong.github.io/algo/di-yi-zhan-da78c/shou-ba-sh-48c1d/er-wei-shu-150fb/
    List<Integer> spiralOrder(int[][] matrix) {
        List<Integer> res = new ArrayList<>();
        int m = matrix.length, n = matrix[0].length, low = 0, up = m, l = 0, r = n, i = 0, j = 0;
        while (res.size() < m * n) {
            i = l;
            if (low <= up) {
                while (i <= r) {
                    res.add(matrix[low][i++]);
                }
            }
            low++;
            i = low;
            if (l <= r) {
                while (i <= up) {
                    res.add(matrix[i++][r]);
                }
            }
            r--;
            i = r;
            if (low <= up) {
                while (i >= l) {
                    res.add(matrix[up][i--]);
                }
            }
            up--;
            i = up;
            if (l <= r) {
                while (i >= low) {
                    res.add(matrix[i--][l]);
                }
            }
            l++;
        }
        System.out.println(res);
        return res;

    }

    //
//    //59. 螺旋矩阵 II
//    public int[][] generateMatrix(int n) {
//
//    }
//
//    //73. 矩阵置零
    public void setZeroes(int[][] matrix) {
        int m = matrix.length, n = matrix[0].length;
        boolean row0 = Arrays.stream(matrix[0]).anyMatch(e -> e == 0);
        boolean col0 = false;
        for (int i = 0; i < m; i++) {
            if (matrix[i][0] == 0) {
                col0 = true;
                break;
            }
        }
        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                if (matrix[i][j] == 0) {
                    matrix[0][j] = 0;
                    matrix[i][0] = 0;
                }
            }
        }
        MyUtile.disMap(matrix);
        for (int i = 1; i < m; i++) {
            if (matrix[i][0] == 0) {
                Arrays.fill(matrix[i], 0);
            }
        }
        MyUtile.disMap(matrix);
        for (int j = 1; j < n; j++) {
            if (matrix[0][j] == 0) {
                for (int i = 0; i < m; i++) {
                    matrix[i][j] = 0;
                }
            }
        }
        if (row0) {
            Arrays.fill(matrix[0], 0);

        }
        if (col0) {
            for (int i = 0; i < m; i++) {
                matrix[i][0] = 0;
            }
        }
        MyUtile.disMap(matrix);


//        boolean col0 = false, row0 = false;
//        int m = matrix.length, n = matrix[0].length, i, j;
//        for (i = 0; i < m && matrix[i][0] != 0; i++) ;
//        col0 = i != m;
//        for (i = 0; i < n && matrix[0][i] != 0; i++) ;
//        row0 = i != n;
//
//        for (i = 1; i < m; i++) {
//            for (j = 1; j < n; j++) {
//                if (matrix[i][j] == 0) {
//                    matrix[0][j] = matrix[i][0] = 0;
//                }
//
//            }
//
//        }
//        for (i = 1; i < m; i++) {
//            for (j = 1; j < n; j++) {
//                if (matrix[0][j] == 0 || matrix[i][0] == 0) {
//                    matrix[i][j] = 0;
//                }
//            }
//        }
//
//        if (col0) {
//            for (i = 0; i < m; i++) {
//                matrix[i][0] = 0;
//            }
//
//        }
//        if (row0) {
//            for (i = 0; i < n; i++) {
//                matrix[0][i] = 0;
//            }
//
//        }


    }

    //
//
//    //    128. 最长连续序列
    public int longestConsecutive(int[] nums) {
        Map<Integer, Integer> map = new HashMap<>();
        for (int num : nums) {
            map.put(num, 0);
        }
        int max = 0;
        for (Map.Entry<Integer, Integer> entry : map.entrySet()) {
            Integer k = entry.getKey();
            Integer v = entry.getValue();
            int tmax = 1, tk = k;
            while (map.containsKey(tk - 1) && map.get(tk - 1) == 0) {
                map.put(tk - 1, 1);
                tk--;
                tmax++;
            }
            tk = k;
            while (map.containsKey(tk + 1) && map.get(tk + 1) == 0) {
                map.put(tk + 1, 1);
                tk++;
                tmax++;
            }
            max = Math.max(max, tmax);
//            max.set(Math.max(max.get(), tmax));
        }
//        System.out.println(max.get());
        return max;

    }

    //
//    //172. 阶乘后的零 https://labuladong.github.io/algo/di-san-zha-24031/shu-xue-yu-659f1/jiang-lian-ae367/
    public int trailingZeroes(int n) {
        int res = 0, d = 5;
        while (d <= n) {
            res += n / d;
            d *= 5;
        }
        return res;
    }

    //    //179. 最大数
    public String largestNumber(int[] nums) {
        List<Integer> list = Arrays.stream(nums).boxed().collect(Collectors.toList());
        list.sort((x, y) -> {
            int sx = 10, sy = 10;
            while (sx <= x) sx *= 10;
            while (sy <= y) sy *= 10;
            return (sx * y + x) - (sy * x + y);
        });
        System.out.println(list);

        StringBuilder sb = new StringBuilder();
        list.forEach(sb::append);
        return Objects.equals(sb.charAt(0), '0') ? "0" : sb.toString();
    }

    //
//    //189. 轮转数组
    public void rotate(int[] nums, int k) {
        k %= nums.length;
        reverse(nums, 0, nums.length - 1);
        reverse(nums, 0, k - 1);
        reverse(nums, k, nums.length - 1);
    }

    public void reverse(int[] nums, int l, int r) {
        while (l < r) {
            swap(nums, l++, r--);
        }
    }

    //
//    //204. 计数质数 高效排除法
//    public int countPrimes(int n) {
//        boolean[] res = new boolean[n + 1];
//        Arrays.fill(res, true);
//        for (int i = 2; i * i < n; i++) {
//            if (res[i]) {
//                for (int j = i * i; j < n; j += i) {
//                    res[j] = false;
//                }
//            }
//        }
//        int c = 0;
//        for (int i = 2; i < res.length - 1; i++) {
//            if (res[i]) {
//                System.out.println(i);
//                c++;
//            }
//        }
//        System.out.println(c);
//        return c;
//
//    }
//
//    //238. 除自身以外数组的乘积
    public int[] productExceptSelf(int[] nums) {
        int n = nums.length;
        int[] res = new int[n];
        Arrays.fill(res, 1);
//        res[0] = 1;
        for (int i = 1; i < n; i++) {
            res[i] = nums[i - 1] * res[i - 1];
        }
        int[] res2 = new int[n];
        Arrays.fill(res2, 1);
        for (int i = n - 2; i >= 0; i--) {
            res2[i] = nums[i + 1] * res2[i + 1];
            res[i] *= res2[i];
        }
        MyUtile.dis(res);
        return res;
    }

    //    //378. 有序矩阵中第 K 小的元素
    public int kthSmallest(int[][] matrix, int k) {
        PriorityQueue<Pair<Integer, Pair<Integer, Integer>>> heap = new PriorityQueue<>((o1, o2) -> o1.getKey() - o2.getKey());
        int n = matrix.length, m = matrix[0].length;
        for (int i = 0; i < n; i++) {
            heap.offer(new Pair<>(matrix[i][0], new Pair<>(i, 0)));
        }
        while (k > 1) {
            int i = heap.peek().getValue().getKey(), j = heap.peek().getValue().getValue() + 1;
            if (j < m) {
                heap.offer(new Pair<>(matrix[i][j], new Pair<>(i, j)));
            }
            heap.poll();
            k--;
        }
        return heap.peek().getKey();

    }

    //
//    //454. 四数相加 II
    public int fourSumCount(int[] nums1, int[] nums2, int[] nums3, int[] nums4) {
        Map<Integer, Integer> map = new HashMap<>();
        for (int num1 : nums1) {
            for (int num2 : nums2) {
                map.put(num1 + num2, map.getOrDefault(num1 + num2, 0) + 1);
            }
        }
        int res = 0;
        for (int num3 : nums3) {
            for (int num4 : nums4) {
                if (map.containsKey(-num3 - num4)) {
                    res += map.get(-num3 - num4);
                }
            }
        }
        return res;
    }

    //
//
//    //26. 删除有序数组中的重复项
    public int removeDuplicates(int[] nums) {
        int slow = 0, fast = 0;
        while (fast < nums.length) {
            if (fast == 0 || nums[fast] != nums[fast - 1]) {
                nums[slow++] = nums[fast];
            }
            fast++;
        }
        MyUtile.dis(nums);
        return slow + 1;
    }

    //
//    //66. 加一
//    public int[] plusOne(int[] digits) {
//        int n = digits.length;
//        int carry = 0;
//
//        if (digits[n - 1] + 1 == 10) carry = 1;
//        digits[n - 1] = (digits[n - 1] + 1) % 10;
//        for (int i = n - 2; i >= 0; i--) {
//            if (digits[i] + carry >= 10) {
//                digits[i] = (digits[i] + carry) % 10;
//                carry = 1;
//            } else {
//                digits[i] += carry;
//                carry = 0;
//            }
//        }
//        if (carry == 1) {
//            int[] res = new int[n + 1];
//            res[0] = 1;
//            for (int i = 1; i < res.length; i++) {
//                res[i] = digits[i - 1];
//            }
//            return res;
//        }
//        return digits;
//
//    }
//
//    //    88. 合并两个有序数组 逆向指针
    public void merge(int[] nums1, int m, int[] nums2, int n) {
        int p1 = m - 1, p2 = n - 1, ind = m + n - 1;
        while (p1 != -1 || p2 != -1) {
            if (p2 == -1) {
                break;
            } else if (p1 == -1) {
                nums1[ind--] = nums2[p2--];
            } else if (nums1[p1] >= nums2[p2]) {
                nums1[ind--] = nums1[p1--];
            } else if (nums1[p1] < nums2[p2]) {
                nums1[ind--] = nums2[p2--];
            }
        }
        MyUtile.dis(nums1);

    }

    //
//    //118. 杨辉三角
//    public List<List<Integer>> generate(int numRows) {
//        List<List<Integer>> res = new ArrayList<>(numRows);
//        int[][] tmpres = new int[numRows][numRows];
//        for (int i = 0; i < numRows; i++) {
//            for (int j = 0; j <= i; j++) {
//                tmpres[i][j] = j == 0 || i == j ? 1 : tmpres[i - 1][j] + tmpres[i - 1][j - 1];
//            }
//            res.add(Arrays.stream(tmpres[i]).boxed().collect(Collectors.toList()).subList(0, i + 1));
//        }
//        System.out.println(res);
//        return res;
//
//
//    }
//
//    //190. 颠倒二进制位
    public int reverseBits(int n) {
        int res = 0;
        for (int i = 0; i < 32; i++) {
            res <<= 1;
            res |= (n & 1);
            n >>= 1;
        }
        System.out.println(res);
        return res;
    }

    //    //191. 位1的个数
    public int hammingWeight(int n) {
        int res = -3;
        while (n != 0) {
            n &= (n - 1);
            res++;
            System.out.println(n + ":" + Integer.toBinaryString(res));
        }
//        System.out.println(Integer.toBinaryString(res));
//        res >>= 5;
//        System.out.println(res);
//        System.out.println(Integer.toBinaryString(res));
//        System.out.println(res >> 1);
        return res;

    }

    //    //202. 快乐数
    public boolean isHappy(int n) {
        int slow = n, fast = getNext(n);
        while (fast != 1 && fast != slow) {
            slow = getNext(slow);
            fast = getNext(getNext(fast));
        }
        return fast == 1;
    }

    public int getNext(int n) {
        int res = 0;
        while (n != 0) {
            int t = n % 10;
            res += t * t;
            n /= 10;
        }
        return res;
    }

    ////    217. 存在重复元素
    public boolean containsDuplicate(int[] nums) {
        Set<Integer> set = new HashSet<>();
        for (int num : nums) {
            if (!set.add(num)) {
                return true;
            }
        }
        return false;
    }

    //    //268. 丢失的数字 异或运算
    public int missingNumber(int[] nums) {
        int res = 0, n = nums.length;
        res ^= n;
        for (int i = 0; i < nums.length; i++) {
            res ^= (i ^ nums[i]);
        }
        return res;
    }

    //
//    //350. 两个数组的交集 II
    public int[] intersect(int[] nums1, int[] nums2) {
        int p1 = 0, p2 = 0;
        Arrays.sort(nums1);
        Arrays.sort(nums2);
        List<Integer> list = new ArrayList<Integer>();
        while (p1 < nums1.length && p2 < nums2.length) {
            if (nums1[p1] == nums2[p2]) {
                list.add(nums1[p1++]);
                p2++;
            } else if (nums1[p1] < nums2[p2]) {
                p1++;
            } else if (nums1[p1] > nums2[p2]) {
                p2++;
            }
        }
        return list.stream().mapToInt(Integer::intValue).toArray();
    }

    //    //    350. 两个数组的交集 II 第二种
    public int[] intersect2(int[] nums1, int[] nums2) {
        Map<Integer, Integer> map = new HashMap<>();
        List<Integer> list = new ArrayList<Integer>();
        for (int i = 0; i < nums1.length; i++) {
            map.put(nums1[i], map.getOrDefault(nums1[i], 0) + 1);
        }
        for (int i = 0; i < nums2.length; i++) {
            if (map.containsKey(nums2[i]) && map.get(nums2[i]) > 0) {
                list.add(nums2[i]);
                map.compute(nums2[i], (k, v) -> v - 1);
            }
        }
        return list.stream().mapToInt(Integer::intValue).toArray();

    }

    //
//    //326. 3 的幂
    public boolean isPowerOfThree(int n) {
        while (n > 0 && n % 3 == 0) {
            n /= 3;
        }
        return n == 1;


    }

    //    //42. 接雨水
    public int trap(int[] height) {

        int l = 0, r = height.length - 1, lm = -99999, rm = -99999, res = 0;
        while (l < r) {
            lm = Math.max(lm, height[l]);
            rm = Math.max(rm, height[r]);
            if (lm < rm) {
                res += lm - height[l++];
            } else {
                res += rm - height[r--];
            }
        }
        return res;

    }

    //
//    //    11. 盛最多水的容器
    public int maxArea(int[] height) {
        int n = height.length, res = 0, l = 0, r = n - 1, lm = -9999, rm = -9999;
        while (l < r) {
            lm = Math.max(lm, height[l]);
            rm = Math.max(rm, height[r]);
            if (lm < rm) {
                res = Math.max(res, (r - l) * lm);
                l++;
            } else {
                res = Math.max(res, (r - l) * rm);
                r--;
            }
        }
        return res;
    }

    //
//    //41. 缺失的第一个正数
    public int firstMissingPositive(int[] nums) {
        int n = nums.length;
        for (int i = 0; i < n; i++) {
            if (nums[i] <= 0) {
                nums[i] = n + 1;
            }
        }
        for (int i = 0; i < n; i++) {
            int ind = Math.abs(nums[i]) - 1;
            if (ind < n && nums[ind] > 0) {
                nums[ind] = -nums[ind];
            }
        }
        for (int i = 0; i < n; i++) {
            if (nums[i] > 0) {
                return i + 1;
            }
        }
        return n + 1;
    }

    ////239. 滑动窗口最大值 单调队列
    public int[] maxSlidingWindow(int[] nums, int k) {
        Deque<Integer> q = new LinkedList<>();
        List<Integer> res = new ArrayList<>();
        for (int i = 0; i < k; i++) {
            while (!q.isEmpty() && q.peekLast() < nums[i]) {
                q.pollLast();
            }
            q.offerLast(nums[i]);
        }
        res.add(q.peekFirst());

        int l = 0, r = k, n = nums.length;
        while (r < n) {
            if (q.peek() == nums[l]) {
                q.pollFirst();
            }
            while (!q.isEmpty() && q.peekLast() < nums[r]) {
                q.pollLast();
            }
            q.offerLast(nums[r]);
            res.add(q.peekFirst());
            l++;
            r++;
        }
        System.out.println(res);
        return res.stream().mapToInt(Integer::intValue).toArray();
    }

    //
//    //134. 加油站
    public int canCompleteCircuit(int[] gas, int[] cost) {
        int res = 0, sum = 0, mincos = 0;
        for (int i = 0; i < gas.length; i++) {
            sum += (gas[i] - cost[i]);
            if (sum < mincos) {
                mincos = sum;
                res = i + 1;
            }
        }
        return sum >= 0 ? res : -1;
    }

    //
//    //496. 下一个更大元素 I 单调栈
    public int[] nextGreaterElement(int[] nums1, int[] nums2) {
        Stack<Integer> stack = new Stack<>();
        Map<Integer, Integer> map = new HashMap<>();
        int n = nums2.length;
        for (int i = n - 1; i >= 0; i--) {
            if (stack.isEmpty()) {
                map.put(nums2[i], -1);
            } else {
                while (!stack.isEmpty() && stack.peek() <= nums2[i]) {
                    stack.pop();
                }
                if (stack.isEmpty()) {
                    map.put(nums2[i], -1);
                } else {
                    map.put(nums2[i], stack.peek());
                }
            }
            stack.push(nums2[i]);

        }
        List<Integer> res = new ArrayList<>();
        for (int i = 0; i < nums1.length; i++) {
            res.add(map.get(nums1[i]));
        }
        System.out.println(res);
        return res.stream().mapToInt(Integer::intValue).toArray();

    }

    //
//    //503. 下一个更大元素 II
    public int[] nextGreaterElements(int[] nums) {
        List<Integer> list = new ArrayList<>();
        list.addAll(Arrays.stream(nums).boxed().collect(Collectors.toList()));
        list.addAll(Arrays.stream(nums).boxed().collect(Collectors.toList()));
        Stack<Integer> stack = new Stack<>();
        List<Integer> res = new ArrayList<>();
        int[] newNums = list.stream().mapToInt(Integer::intValue).toArray();
        for (int i = newNums.length - 1; i >= 0; i--) {
            while (!stack.isEmpty() && stack.peek() <= newNums[i]) {
                stack.pop();
            }
            res.add(stack.isEmpty() ? -1 : stack.peek());
            stack.push(newNums[i]);
        }
        Collections.reverse(res);
        return res.subList(0, nums.length).stream().mapToInt(Integer::intValue).toArray();
    }

    //
//    //739. 每日温度
    public int[] dailyTemperatures(int[] temperatures) {
        Stack<Pair<Integer, Integer>> stack = new Stack<>();
        List<Integer> res = new ArrayList<>();
        for (int i = temperatures.length - 1; i >= 0; i--) {
            while (!stack.isEmpty() && stack.peek().getKey() <= temperatures[i]) {
                stack.pop();
            }
            res.add(stack.isEmpty() ? 0 : stack.peek().getValue() - i);
            stack.push(new Pair<>(temperatures[i], i));
        }
        Collections.reverse(res);
        return res.stream().mapToInt(Integer::intValue).toArray();
    }

//    public static void main(String[] args) {
//        Numbers numbers = new Numbers();
//        int[] nums = new int[]{1, 2, 0};
//
//    }


}
