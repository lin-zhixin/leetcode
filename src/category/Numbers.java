package category;

//数字相关题目

import javafx.util.Pair;

import java.util.*;
import java.util.concurrent.locks.LockSupport;
import java.util.concurrent.locks.ReentrantLock;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;
//import org.apache.commons.collections.CollectionUtils;

public class Numbers {


    //8. 字符串转换整数 (atoi)
    public int myAtoi(String s) {
        s = s.trim();
        if (s.length() == 0) {
            return 0;
        }

        long res = 0;
        int i = 0, n = s.length();
        int sign = 1;
        if (s.charAt(i) == '-') {
            sign = -1;
            i++;
        } else if (s.charAt(i) == '+') {
            i++;
        }
        while (i < n && s.charAt(i) >= '0' && s.charAt(i) <= '9') {
            res = res * 10 + s.charAt(i) - '0';
            if (res > Integer.MAX_VALUE) {
                return sign == 1 ? Integer.MAX_VALUE : Integer.MIN_VALUE;
            }
            i++;
        }
        return (int) res * sign;

    }

    public int myAtoi2(String str) {
//        别人的笔记
        int n = str.length();
        int i = 0;
        // 记录正负号
        int sign = 1;
        // 用 long 避免 int 溢出
        long res = 0;
        // 跳过前导空格
        while (i < n && str.charAt(i) == ' ') {
            i++;
        }
        if (i == n) {
            return 0;
        }

        // 记录符号位
        if (str.charAt(i) == '-') {
            sign = -1;
            i++;
        } else if (str.charAt(i) == '+') {
            i++;
        }
        if (i == n) {
            return 0;
        }

        // 统计数字位
        while (i < n && '0' <= str.charAt(i) && str.charAt(i) <= '9') {
            res = res * 10 + str.charAt(i) - '0';
            if (res > Integer.MAX_VALUE) {
                break;
            }
            i++;
        }
        // 如果溢出，强转成 int 就会和真实值不同
        if ((int) res != res) {
            return sign == 1 ? Integer.MAX_VALUE : Integer.MIN_VALUE;
        }
        return (int) res * sign;
    }


//    soul面试：233. 数字 1 的个数

    public int countDigitOne(int n) {
        //        枚举每一位可能出现1的个数，左边个数+右边个数，左边就是左边的大小，右边就是根据右边的数字大小决定，如果数字在1xxx~199...之间就是r-1xxx+1,小于1xxx就是0，大于199...就是1xx个
        int len = (n + "").length();
        //        System.out.println(len);
        int res = 0;

        for (int i = 0; i < len; i++) {
            int l = n / myPow(10, i + 1);
            int r = n % myPow(10, i + 1);
            int lsum = l * myPow(10, i);
            int rsum = Math.min(myPow(10, i), Math.max(0, r - myPow(10, i) + 1));
            res += (lsum + rsum);
        }
        return res;
    }

    public int myPow(int n, int k) {
        int res = 1;
        while (k > 0) {
            int t = (k % 2);
            if (t == 1) {
                res *= n;
            }
            n *= n;
            k /= 2;
        }
        //        System.out.println(res);
        return res;
    }

    //1. 两数之和
    public int[] twoSum(int[] nums, int target) {
        List<Pair<Integer, Integer>> numbers = new ArrayList<>();
        for (int i = 0; i < nums.length; i++) {
            numbers.add(new Pair<>(nums[i], i));
        }
        numbers.sort(((o1, o2) -> o1.getKey() - o2.getKey()));
        int n = numbers.size();
        int l = 0, r = n - 1;
        List<Integer> res = new ArrayList<>();
        while (l < r) {
            int left = numbers.get(l).getKey(), right = numbers.get(r).getKey();
            int sum = left + right;
            if (sum == target) {
                res.add(numbers.get(l).getValue());
                res.add(numbers.get(r).getValue());
                while (l < r && numbers.get(l).getKey() == left) l++;
                while (l < r && numbers.get(r).getKey() == right) r--;
            } else if (sum < target) {
                while (l < r && numbers.get(l).getKey() == left) l++;
            } else if (sum > target) {
                while (l < r && numbers.get(r).getKey() == right) r--;
            }
        }
        System.out.println(res);
        return res.stream().mapToInt(Integer::intValue).toArray();
    }

    //    合并区间
    public int[][] merge(int[][] intervals) {
        Arrays.sort(intervals, (a, b) -> a[0] - b[0]);
        int l = intervals[0][0], r = intervals[0][1];
        List<int[]> res = new ArrayList<>();
        for (int i = 1; i < intervals.length; i++) {
            if (intervals[i][0] <= r && intervals[i][1] > r) {
                r = intervals[i][1];
                continue;
            }
            if (intervals[i][0] > r) {
                res.add(new int[]{l, r});
                l = intervals[i][0];
                r = intervals[i][1];
            }
        }
        res.add(new int[]{l, r});
        System.out.println(res);
        return res.toArray(new int[res.size()][]);
    }

    //57. 插入区间
    public int[][] insert(int[][] intervals, int[] newInterval) {
        int n = intervals.length, i;
        int[][] ints = new int[n + 1][2];
        for (i = 0; i < n; i++) {
            ints[i][0] = intervals[i][0];
            ints[i][1] = intervals[i][1];
        }
        ints[i][0] = newInterval[0];
        ints[i][1] = newInterval[1];

        Arrays.sort(ints, (o1, o2) -> o1[0] == o2[0] ? o2[1] - o1[1] : o1[0] - o2[0]);

        int l = ints[0][0], r = ints[0][1];
        List<int[]> res = new ArrayList<>();
        for (int j = 0; j < ints.length; j++) {
            if (ints[j][0] <= r && r <= ints[j][1]) {
                r = ints[j][1];
            } else if (r < ints[j][0]) {
                res.add(new int[]{l, r});
                l = ints[j][0];
                r = ints[j][1];
            }
//            System.out.println(l + " :" + r);
        }
        res.add(new int[]{l, r});
//        System.out.println(res);
        return res.toArray(new int[res.size()][]);
    }


    //1288. 删除被覆盖区间
    public int removeCoveredIntervals(int[][] intervals) {
        Arrays.sort(intervals, (a, b) -> a[0] == b[0] ? b[1] - a[1] : a[0] - b[0]);
        int l = intervals[0][0], r = intervals[0][1];
        int cnt = 0;
        for (int i = 1; i < intervals.length; i++) {
            if (intervals[i][0] >= l && intervals[i][1] <= r) {
                cnt++;
            }
            if (intervals[i][1] > r) {
                l = intervals[i][0];
                r = intervals[i][1];
            }
        }
        System.out.println(cnt);
        return intervals.length - cnt;
    }

    //    986. 区间列表的交集
    public int[][] intervalIntersection(int[][] firstList, int[][] secondList) {
        int al, ar, bl, br, i = 0, j = 0;
        List<int[]> res = new ArrayList<>();
        while (i < firstList.length && j < secondList.length) {
            al = firstList[i][0];
            ar = firstList[i][1];
            bl = secondList[j][0];
            br = secondList[j][1];
            if (bl <= ar && br >= al) {
                res.add(new int[]{Math.max(al, bl), Math.min(ar, br)});
            }
            if (ar > br) {
                j++;
            } else {
                i++;
            }
        }
        return res.toArray(new int[res.size()][]);
    }

    //    面試題：topk问题 215. 数组中的第K个最大元素
    public int findKthLargest3(int[] nums, int k) {
//        LockSupport.park();
//        ReentrantLock
        return qsortSelect(nums, 0, nums.length - 1, nums.length - k);
    }

    public int qsortSelect(int[] nums, int l, int r, int k) {
        if (l < r) {
            int p = topKPart(nums, l, r, k);
            if (p == k) {
                return nums[p];
            } else if (p < k) {
                return qsortSelect(nums, p + 1, r, k);
            } else if (p > k) {
                return qsortSelect(nums, l, p - 1, k);
            }
        }
        return nums[l];
    }

    public int topKPart(int[] nums, int l, int r, int k) {
        int p = nums[l];
        while (l < r) {
            while (l < r && nums[r] >= p) r--;
            nums[l] = nums[r];
            while (l < r && nums[l] <= p) l++;
            nums[r] = nums[l];
        }
        nums[l] = p;
        return l;
    }

    //33. 搜索旋转排序数组 二分
    public int search2(int[] nums, int target) {
        int n = nums.length, l = 0, r = n - 1;
        while (l < r) {
            int mid = l + (r - l) / 2;
            if (target == nums[mid]) {
                return mid;
            }
//            如果mid在左侧那么左边是有序序列，如果 target属于右侧的话就l = mid + 1;否则就 r = mid - 1;
//            mid在右侧同理
            if (nums[mid] >= nums[0]) {
                if (target > nums[mid] || target < nums[0]) {
                    l = mid + 1;
                } else {
                    r = mid - 1;
                }
            } else {
                if (target > nums[mid] && target < nums[0]) {
                    l = mid + 1;
                } else {
                    r = mid - 1;
                }
            }
        }
        return nums[l] == target ? l : -1;

    }

    //    153. 寻找旋转排序数组中的最小值
    public int findMin1(int[] nums) {
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

    //    改编为找最大 153. 寻找旋转排序数组中的最小值
    public int findMin3(int[] nums) {
        int l = 0, r = nums.length;
        while (l < r) {
            int mid = l + (r - l) / 2;
            if (nums[mid] < nums[l]) {
                r = mid;
            } else if (nums[mid] > nums[l]) {
                l = mid;
            } else {
                l++;
            }
        }
        return nums[l];

    }

//---------------2023.8.23之前：

    //    540. 有序数组中的单一元素
    public int singleNonDuplicate(int[] nums) {

        int l = 0, r = nums.length;
        while (l < r) {
            int mid = l + (l + r) / 2;
            if (nums[mid] == nums[mid ^ 1]) {
                l = mid + 1;
            } else {
                r = mid;
            }
        }
        return nums[l];
    }

    //    224. 基本计算器
    public int calculate(String s) {
        Deque<Character> q = new LinkedList<>();
        for (int i = 0; i < s.length(); i++) {
            if (s.charAt(i) != ' ' && q.offer(s.charAt(i))) ;
        }
        return calculate(q);
    }

    public int calculate(Deque<Character> q) {
        Stack<Integer> stack = new Stack<>();
        Character c, sign = '+';
        Integer num = 0;
        while (!q.isEmpty()) {
            c = q.poll();
            if (Character.isDigit(c)) {
                num = num * 10 + (c - '0');
            }
            if (c == '(') {
//                stack.push(calculate(q));
                num = calculate(q);
            }

            if (!Character.isDigit(c) || q.isEmpty()) {
                switch (sign) {
                    case '+': {
                        stack.push(num);
                        break;
                    }
                    case '-': {
                        System.out.println(-num);
                        stack.push(-num);
                        break;
                    }
                    case '*': {
                        stack.push(stack.pop() * num);
                        break;
                    }
                    case '/': {
                        stack.push(stack.pop() / num);
                        break;
                    }
                }
                sign = c;
                num = 0;
            }
            if (c == ')') {
                System.out.println(stack);
                break;
            }
        }
        return stack.stream().mapToInt(Integer::intValue).sum();
    }

    //43. 字符串相乘
    public String multiply(String num1, String num2) {
        int m = num1.length(), n = num2.length();
        int[] res = new int[m + n], arr1 = new int[m], arr2 = new int[n];
        for (int i = 0; i < m; i++) {
            arr1[i] = num1.charAt(i) - '0';
        }
        for (int i = 0; i < n; i++) {
            arr2[i] = num2.charAt(i) - '0';
        }
        for (int i = m - 1; i >= 0; i--) {
            for (int j = n - 1; j >= 0; j--) {
                int t = arr1[i] * arr2[j];
//                要加的下标位置十位数对应的是i + j，个位数对应的是i + j + 1,
                int sum = res[i + j + 1] + t;
                res[i + j + 1] = sum % 10;
                res[i + j] += sum / 10;//进位的事情会由下一次循环的时候处理，最后得出的结果是进位过的，但是中间过程res的某一个位置可能出现超过9的数。
            }
        }
        int k = 0;
        while (k < res.length && res[k] == 0) k++;
        StringBuilder sb = new StringBuilder();
        for (int i = k; i < res.length; i++) {
            sb.append(res[i]);
        }
        return "".equals(sb.toString()) ? "0" : sb.toString();
    }

    //659. 分割数组为连续子序列
    public boolean isPossible(int[] nums) {
        Map<Integer, Integer> freq = new HashMap<>();
        Map<Integer, Integer> need = new HashMap<>();
        Arrays.stream(nums).forEach(e -> freq.put(e, freq.getOrDefault(e, 0) + 1));

        for (int num : nums) {
            if (freq.get(num) == 0) {
                continue;
            }
            if (need.containsKey(num) && need.get(num) > 0) {
                freq.put(num, freq.get(num) - 1);
                need.put(num, need.get(num) - 1);
                need.put(num + 1, need.getOrDefault(num + 1, 0) + 1);
            } else if (freq.containsKey(num) && freq.get(num) > 0 && freq.containsKey(num + 1) && freq.get(num + 1) > 0 && freq.containsKey(num + 2) && freq.get(num + 2) > 0) {
                freq.put(num, freq.get(num) - 1);
                freq.put(num + 1, freq.get(num + 1) - 1);
                freq.put(num + 2, freq.get(num + 2) - 1);
                need.put(num + 3, need.getOrDefault(num + 3, 0) + 1);
            } else {
                return false;
            }
        }
        return true;
    }


    //    作者：LeetCode-Solution
//    链接：https://leetcode.cn/problems/sqrtx/solution/x-de-ping-fang-gen-by-leetcode-solution/
//    来源：力扣（LeetCode）
//    著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

    // 重要：215. 数组中的第K个最大元素 topK问题
    public int findKthLargest(int[] nums, int k) {
        shuffle(nums);
        int n = nums.length, l = 0, r = n - 1;
        k = n - k;
        while (l <= r) {
            int p = part(nums, l, r);
            if (p == k) {
                return nums[p];
            } else if (p < k) {
                l = p + 1;
            } else if (p > k) {
                r = p - 1;
            }
        }
        return -1;
    }

    //    堆排序做法
    public int findKthLargest2(int[] nums, int k) {
        int n = nums.length;
        buildHeap2(nums);
//        k++;
        for (int i = 0; i < k - 1; i++) {
            swap(nums, 0, n - 1 - i);
            down2(nums, 0, n - 1 - i);
        }
        return nums[0];
    }

    public void buildHeap2(int[] nums) {
        int n = nums.length;
        for (int i = n / 2; i >= 0; i--) {
            down2(nums, i, n);
        }
    }

    public void down2(int[] nums, int p, int end) {
        int c = p * 2 + 1;
        while (c < end) {
            c = c + 1 < end && nums[c + 1] > nums[c] ? c + 1 : c;
            if (nums[p] < nums[c]) {
                swap(nums, p, c);
            }
            p = c;
            c = p * 2 + 1;
        }
    }


    //    912. 排序数组
    public int[] sortArray1(int[] nums) {
        shuffle(nums);
        qsort(nums, 0, nums.length - 1);
        return nums;
    }

    public void qsort(int[] nums, int l, int r) {
        if (l < r) {
            int p = part(nums, l, r);
            qsort(nums, l, p - 1);
            qsort(nums, p + 1, r);
        }
    }

    public void shuffle(int[] nums) {
        Random random = new Random();
        int n = nums.length;
        for (int i = 0; i < n; i++) {
            int t = i + random.nextInt(n - i);
            swap(nums, i, t);
        }
    }

    public int part(int[] list, int l, int r) {
        int mid = list[l];
        while (l < r) {
            while (l < r && list[r] >= mid) r--;
            list[l] = list[r];
            while (l < r && list[l] <= mid) l++;
            list[r] = list[l];
        }
        list[l] = mid;
        return l;
    }

    //    堆排序
    public int[] sortArray(int[] nums) {
        int n = nums.length;
        buildHeap(nums);
        for (int i = n - 1; i >= 0; i--) {
            swap(nums, 0, i);
            down(nums, 0, i);
        }
        return nums;
    }

    public void buildHeap(int[] nums) {
        int n = nums.length;
        for (int i = n / 2; i >= 0; i--) {
            down(nums, i, n);
        }
    }

    //    调整堆
    public void down(int[] nums, int p, int end) {
        int n = nums.length, c = p * 2 + 1;
        while (c < end) {
            c = c + 1 < end && nums[c + 1] > nums[c] ? c + 1 : c;
            if (nums[p] < nums[c]) {
                swap(nums, p, c);
            }
            p = c;
            c = p * 2 + 1;
        }
    }


    //    69. x 的平方根 重要理论知识：一个整数的平方根肯定不会超过它自己的一半 ，因为是要求整数部分 因此可以直接int除，整体的流程就是左右两边取中点，大了右边左移1，小了左边右移1
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

    //31. 下一个排列 根据数字组合的特征解决
    // 1.如果序列全部降序说明这个序列已经是最大的了
    // 2.从后往前如果升序到某一个地方突然某一个数k降下去说明这个序列不是最大的还可以提升 并且使用k与后面的序列中的一个大于k的最小值交换然后后面重排为升序就是下一个序列
    public void nextPermutation(int[] nums) {
        int n = nums.length, i, j;
        for (i = n - 2; i >= 0 && nums[i] >= nums[i + 1]; i--) ;
//        System.out.println(nums[i]);
        if (i >= 0) {
            for (j = n - 1; j > i && nums[j] <= nums[i]; j--) ;
//        System.out.println(nums[j]);
            j = j < 0 ? n - 1 : j;
            if (nums[j] != nums[i]) {
                nums[j] = nums[j] ^ nums[i];
                nums[i] = nums[j] ^ nums[i];
                nums[j] = nums[j] ^ nums[i];
            }
        }
        List<Integer> res = Arrays.stream(nums).boxed().collect(Collectors.toList());
        List<Integer> tmp = res.subList(i + 1, n);
        Collections.reverse(tmp);
        System.out.println(tmp);
        res = res.subList(0, i + 1);
        res.addAll(tmp);
//        nums=res.stream().mapToInt(Integer::intValue).toArray();
        for (int k = 0; k < n; k++) {
            nums[k] = res.get(k);
        }
        System.out.println(res);
    }

    //    全排列
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
        int l = 0, r = nums.length - 1;
        while (l <= r) {
            int mid = (l + r) / 2;
            if (target == nums[mid]) {
                r = mid - 1;
            } else if (target < nums[mid]) {
                r = mid - 1;
            } else if (target > nums[mid]) {
                l = mid + 1;
            }
        }
        return l;
    }

    public int upperBound(int[] nums, int target) {
        int l = 0, r = nums.length - 1;
        while (l <= r) {
            int mid = (l + r) / 2;
            if (nums[mid] == target) {
                l = mid + 1;
            } else if (nums[mid] > target) {
                r = mid - 1;
            } else if (nums[mid] < target) {
                l = mid + 1;
            }
        }
        return l;
    }

//    public int lowerBound(int[] nums, int target) {
//        int l = 0, r = nums.length;
//        while (l < r) {//跳出条件是l==r 也就是[nums.length,nums.length)
//            int mid = (l + r) / 2;
//            if (target == nums[mid]) {
//                r = mid;// 右边是闭区间 此时变成[l,mid)
//            } else if (target < nums[mid]) {
//                r = mid;
//            } else if (target > nums[mid]) {
//                l = mid + 1;
//            }
//        }
//        return l;//正常的话需要加判断nums[l]!= target? -1 : l;
//
//    }

//    public int upperBound(int[] nums, int target) {
//        int l = 0, r = nums.length;
//        while (l < r) {//跳出条件是l==r 也就是[nums.length,nums.length)
//            int mid = (l + r) / 2;
//            if (target == nums[mid]) {
//                l = mid + 1;// 右边是闭区间 此时变成[l,mid)
//            } else if (target < nums[mid]) {
//                r = mid;
//            } else if (target > nums[mid]) {
//                l = mid + 1;
//            }
//        }
//        return l;//正常的话需要加判断nums[l-1]!= target? -1 : l;
//    }

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
        int l = 0, n = nums.length, r = n - 1;
        for (int i = 0; i <= r; ) {//!!!等号 i不要加
            if (nums[i] == 2) {
                while (i < r && nums[r] == 2) {
                    r--;
                }
                MyUtile.swap(nums, i, r);

                r--;
            } else if (nums[i] == 0 && i != l) {
                MyUtile.swap(nums, i, l);
                l++;
            } else {
                i++;//!!!
            }
        }
        MyUtile.dis(nums);

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

    //    338. 比特位计数 剑指 Offer II 003. 前 n 个数字二进制中 1 的个数
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
//    231. 2 的幂

    public boolean isPowerOfTwo(int n) {
        if (n <= 0) return false;
        return (n & (n - 1)) == 0;

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
            return getKthElement2(nums1, nums2, totalLen / 2 + 1);//加一是因为k是第几个 没有包括0下标的
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

    //448. 找到所有数组中消失的数字 利用下标
    public List<Integer> findDisappearedNumbers(int[] nums) {
        int n = nums.length;
        for (int i = 0; i < n; i++) {
            nums[(nums[i] - 1) % n] += n;
        }
        List<Integer> res = new ArrayList<>();
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] <= nums.length && res.add(i + 1)) ;
        }
        System.out.println(res);
        return res;

    }

    //461. 汉明距离
    public int hammingDistance(int x, int y) {
        Integer n = x ^ y;
        System.out.println(n);
        String bit = Integer.toBinaryString(n);
        int res = 0;
        for (int i = 0; i < bit.length(); i++) {
            if (Objects.equals(bit.charAt(i), '1')) res++;
        }
        System.out.println();
        return res;
    }

    //48. 旋转图像  https://labuladong.github.io/algo/di-yi-zhan-da78c/shou-ba-sh-48c1d/er-wei-shu-150fb/
    public void rotate(int[][] matrix) {
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j <= i; j++) {
                if (matrix[i][j] != matrix[j][i]) {
                    matrix[i][j] = matrix[i][j] ^ matrix[j][i];
                    matrix[j][i] = matrix[i][j] ^ matrix[j][i];
                    matrix[i][j] = matrix[i][j] ^ matrix[j][i];
                }
            }
        }
        int l = 0, r = matrix.length - 1;
        while (l < r) {
            for (int i = 0; i < matrix.length; i++) {
                if (matrix[i][l] != matrix[i][r]) {
                    matrix[i][l] = matrix[i][l] ^ matrix[i][r];
                    matrix[i][r] = matrix[i][l] ^ matrix[i][r];
                    matrix[i][l] = matrix[i][l] ^ matrix[i][r];
                }
            }
            l++;
            r--;
        }
//        MyUtile.disMap(matrix);
    }


    //54. 螺旋矩阵  https://labuladong.github.io/algo/di-yi-zhan-da78c/shou-ba-sh-48c1d/er-wei-shu-150fb/
    List<Integer> spiralOrder(int[][] matrix) {
        int m = matrix.length, n = matrix[0].length;
        int upper_bound = m - 1, lower_bound = 0;
        int left_bound = 0, right_bound = n - 1;
        List<Integer> res = new LinkedList<>();

        while (res.size() < m * n) {
            if (lower_bound <= upper_bound) {
                for (int i = left_bound; i <= right_bound; i++) {
                    res.add(matrix[lower_bound][i]);
                }
                lower_bound++;
            }
            if (left_bound <= right_bound) {
                for (int i = lower_bound; i <= upper_bound; i++) {
                    res.add(matrix[i][right_bound]);
                }
                right_bound--;
            }
            if (lower_bound <= upper_bound) {
                for (int i = right_bound; i >= left_bound; i--) {
                    res.add(matrix[upper_bound][i]);
                }
                upper_bound--;
            }
            if (left_bound <= right_bound) {
                for (int i = upper_bound; i >= lower_bound; i--) {
                    res.add(matrix[i][left_bound]);
                }
                left_bound++;
            }
        }
        System.out.println(res);
        return res;
    }

    //59. 螺旋矩阵 II
    public int[][] generateMatrix(int n) {
        int t = 1;
        int[][] res = new int[n][n];
        int lower_bound = 0, upper_bound = n - 1;
        int left_bound = 0, right_bound = n - 1;
        while (t <= n * n) {
            if (lower_bound <= upper_bound) {
                for (int i = left_bound; i <= right_bound; i++) {
                    res[lower_bound][i] = t++;
                }
                lower_bound++;
            }
            if (left_bound <= right_bound) {
                for (int i = lower_bound; i <= upper_bound; i++) {
                    res[i][right_bound] = t++;
                }
                right_bound--;
            }
            if (lower_bound <= upper_bound) {
                for (int i = right_bound; i >= left_bound; i--) {
                    res[upper_bound][i] = t++;
                }
                upper_bound--;
            }
            if (left_bound <= right_bound) {
                for (int i = upper_bound; i >= lower_bound; i--) {
                    res[i][left_bound] = t++;
                }
                left_bound++;
            }
        }
        return res;

    }

    //73. 矩阵置零
    public void setZeroes(int[][] matrix) {
        boolean col0 = false, row0 = false;
        int m = matrix.length, n = matrix[0].length, i, j;
        for (i = 0; i < m && matrix[i][0] != 0; i++) ;
        col0 = i != m;
        for (i = 0; i < n && matrix[0][i] != 0; i++) ;
        row0 = i != n;

        for (i = 1; i < m; i++) {
            for (j = 1; j < n; j++) {
                if (matrix[i][j] == 0) {
                    matrix[0][j] = matrix[i][0] = 0;
                }

            }

        }
        for (i = 1; i < m; i++) {
            for (j = 1; j < n; j++) {
                if (matrix[0][j] == 0 || matrix[i][0] == 0) {
                    matrix[i][j] = 0;
                }
            }
        }

        if (col0) {
            for (i = 0; i < m; i++) {
                matrix[i][0] = 0;
            }

        }
        if (row0) {
            for (i = 0; i < n; i++) {
                matrix[0][i] = 0;
            }

        }


    }


    //    128. 最长连续序列
    public int longestConsecutive(int[] nums) {
        Set<Integer> set = Arrays.stream(nums).boxed().collect(Collectors.toSet());

        int tmplen = 0, num, max = 0;
        for (int i = 0; i < nums.length; i++) {
            num = nums[i];
            if (!set.contains(num - 1)) {
                set.add(num);
                tmplen = 1;

                while (set.contains(num + 1)) {
                    num++;
                    tmplen++;
                }
                max = Math.max(max, tmplen);
            }
        }
        System.out.println(max);
        return max;

    }

    //172. 阶乘后的零 https://labuladong.github.io/algo/di-san-zha-24031/shu-xue-yu-659f1/jiang-lian-ae367/
    public int trailingZeroes(int n) {
        int res = 0, dev = 5, t = n;
        while (t > 0) {
            res += t / 5;
            t /= 5;
        }

        System.out.println(res);
        return res;

    }

    //179. 最大数
    public String largestNumber(int[] nums) {

        List<Integer> list = Arrays.stream(nums).boxed().collect(Collectors.toList());
        list.sort((x, y) -> {
            int sx = 10, sy = 10;
            while (sx <= x) {
                sx *= 10;
            }
            while (sy <= y) {
                sy *= 10;
            }
            return sx * y + x - sy * x - y;
        });
        StringBuilder res = new StringBuilder();
        for (Integer o : list) {
            res.append(o.toString());
        }

        System.out.println(res);
        return Objects.equals(res.charAt(0), '0') ? "0" : res.toString();

    }

    //189. 轮转数组
    public void rotate(int[] nums, int k) {
        k %= nums.length;
        reverse(nums, 0, nums.length - 1);
        reverse(nums, 0, k - 1);
        reverse(nums, k, nums.length - 1);
        MyUtile.dis(nums);
    }

    public void reverse(int[] nums, int l, int r) {
        while (l < r) {
            if (nums[l] != nums[r]) {
                nums[l] = nums[l] ^ nums[r];
                nums[r] = nums[l] ^ nums[r];
                nums[l] = nums[l] ^ nums[r];
            }
            l++;
            r--;
        }
    }

    //204. 计数质数 高效排除法
    public int countPrimes(int n) {
        boolean[] res = new boolean[n + 1];
        Arrays.fill(res, true);
        for (int i = 2; i * i < n; i++) {
            if (res[i]) {
                for (int j = i * i; j < n; j += i) {
                    res[j] = false;
                }
            }
        }
        int c = 0;
        for (int i = 2; i < res.length - 1; i++) {
            if (res[i]) {
                System.out.println(i);
                c++;
            }
        }
        System.out.println(c);
        return c;

    }

    //238. 除自身以外数组的乘积
    public int[] productExceptSelf(int[] nums) {
        int n = nums.length;
        int[] res = new int[n];
        res[0] = 1;
        for (int i = 1; i < n; i++) {
            res[i] = nums[i - 1] * res[i - 1];
        }

        int r = 1;
        for (int i = n - 1; i > 0; i--) {
            res[i] *= r;
            r *= nums[i];
        }
        res[0] = r;
//        MyUtile.dis(res);
        return res;
    }

    //240. 搜索二维矩阵 II
    public boolean searchMatrix(int[][] matrix, int target) {
        int i = 0, j = matrix[0].length - 1;
        while (i >= 0 && i < matrix.length && j >= 0 && j < matrix[i].length) {
            if (matrix[i][j] == target) {
                return true;
            } else if (matrix[i][j] < target) {
                i++;
            } else if (matrix[i][j] > target) {
                j--;
            }
        }
        return false;

    }

    //378. 有序矩阵中第 K 小的元素
    public int kthSmallest(int[][] matrix, int k) {
        PriorityQueue<Pair<Integer, Pair<Integer, Integer>>> heap = new PriorityQueue<>(Comparator.comparingInt(Pair::getKey));
        int n = matrix.length;
        for (int i = 0; i < n; i++) {
            heap.add(new Pair<>(matrix[i][0], new Pair<>(i, 0)));
        }
        k--;
        while (k > 0) {
            Pair<Integer, Pair<Integer, Integer>> top = heap.poll();
            if (top.getValue().getValue() + 1 < n) {
                heap.add(new Pair<>(matrix[top.getValue().getKey()][top.getValue().getValue() + 1], new Pair<>(top.getValue().getKey(), top.getValue().getValue() + 1)));
            }
            k--;
        }
        System.out.println(heap.peek());
        return heap.peek().getKey();
    }

    //454. 四数相加 II
    public int fourSumCount(int[] nums1, int[] nums2, int[] nums3, int[] nums4) {
        Map<Integer, Integer> map = new HashMap<>();
        for (int u : nums1) {
            for (int v : nums2) {
                map.put(u + v, map.getOrDefault(u + v, 0) + 1);
            }
        }
        int res = 0;
        for (int u : nums3) {
            for (int v : nums4) {
                if (map.containsKey(-u - v)) {
                    res += map.get(-u - v);
                }
            }
        }
        return res;


    }


    //26. 删除有序数组中的重复项
    public int removeDuplicates(int[] nums) {
        int slow = 0, fast = 0;
        while (fast < nums.length) {
            if (nums[slow] != nums[fast]) {
                nums[++slow] = nums[fast];
            }
            fast++;
        }
        MyUtile.dis(nums);
        return slow + 1;
    }

    //66. 加一
    public int[] plusOne(int[] digits) {
        int n = digits.length;
        int carry = 0;

        if (digits[n - 1] + 1 == 10) carry = 1;
        digits[n - 1] = (digits[n - 1] + 1) % 10;
        for (int i = n - 2; i >= 0; i--) {
            if (digits[i] + carry >= 10) {
                digits[i] = (digits[i] + carry) % 10;
                carry = 1;
            } else {
                digits[i] += carry;
                carry = 0;
            }
        }
        if (carry == 1) {
            int[] res = new int[n + 1];
            res[0] = 1;
            for (int i = 1; i < res.length; i++) {
                res[i] = digits[i - 1];
            }
            return res;
        }
        return digits;

    }

    //    88. 合并两个有序数组 逆向指针
    public void merge(int[] nums1, int m, int[] nums2, int n) {
        int p1 = m - 1, p2 = n - 1, ind = m + n - 1;
        while (p1 != -1 || p2 != -1) {
            int cur;
            if (p1 == -1) {
                cur = nums2[p2--];
            } else if (p2 == -1) {
                cur = nums1[p1--];
            } else if (nums1[p1] > nums2[p2]) {
                cur = nums1[p1--];
            } else {
                cur = nums2[p2--];
            }
            nums1[ind--] = cur;
        }
        MyUtile.dis(nums1);

    }

    //118. 杨辉三角
    public List<List<Integer>> generate(int numRows) {
        List<List<Integer>> res = new ArrayList<>(numRows);
        int[][] tmpres = new int[numRows][numRows];
        for (int i = 0; i < numRows; i++) {
            for (int j = 0; j <= i; j++) {
                tmpres[i][j] = j == 0 || i == j ? 1 : tmpres[i - 1][j] + tmpres[i - 1][j - 1];
            }
            res.add(Arrays.stream(tmpres[i]).boxed().collect(Collectors.toList()).subList(0, i + 1));
        }
        System.out.println(res);
        return res;


    }

    //190. 颠倒二进制位
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

    //191. 位1的个数
    public int hammingWeight(int n) {
        int res = 0;
        while (n > 0) {
            n &= (n - 1);
            res++;
        }
        System.out.println(res);
        return res;

    }


    //202. 快乐数
    public boolean isHappy(int n) {
        int slow = n, fast = getNext(n);
        while (fast != 1 && slow != fast) {
            slow = getNext(slow);
            fast = getNext(getNext(fast));
        }
        return fast == 1;
    }

    public int getNext(int n) {
        int p = 0;
        while (n > 0) {
            p += (n % 10) * (n % 10);
            n /= 10;
        }
        return p;
    }
//    217. 存在重复元素

    public boolean containsDuplicate(int[] nums) {
        Set<Integer> set = new HashSet<Integer>();
        for (int i = 0; i < nums.length; i++) {
            if (!set.add(nums[i])) {
                return true;
            }

        }
        return false;

    }


    //268. 丢失的数字 异或运算
    public int missingNumber(int[] nums) {
        List<Integer> list = Arrays.stream(nums).boxed().collect(Collectors.toList());
        for (int i = 0; i <= nums.length; i++) {
            list.add(i);
        }
        int res = 0;
        for (Integer num : list) {
            res ^= num;
        }
        return res;
    }

    //350. 两个数组的交集 II
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

    //    350. 两个数组的交集 II 第二种
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

    //326. 3 的幂
    public boolean isPowerOfThree(int n) {
        while (n > 0 && n % 3 == 0) {
            n /= 3;
        }
        return n == 1;

    }


    //42. 接雨水
    public int trap(int[] height) {
        int n = height.length;
        int l = 0, r = n - 1, lmax = Integer.MIN_VALUE, rmax = Integer.MIN_VALUE, res = 0;
        while (l < r) {
            lmax = Math.max(lmax, height[l]);
            rmax = Math.max(rmax, height[r]);
            if (lmax < rmax) {
                res += lmax - height[l++];
            } else {
                res += rmax - height[r--];
            }
        }
        return res;

    }

    //    11. 盛最多水的容器
    public int maxArea2(int[] height) {
        int n = height.length;
        int l = 0, r = n - 1, lmax = Integer.MIN_VALUE, rmax = Integer.MIN_VALUE, res = 0;
        while (l < r) {
            lmax = Math.max(lmax, height[l]);
            rmax = Math.max(rmax, height[r]);
            if (lmax < rmax) {
                res = Math.max(res, (r - l) * lmax);
                l++;
            } else {
                res = Math.max(res, (r - l) * rmax);
                r--;
            }
            System.out.println(res);

        }
        return res;

    }

    //41. 缺失的第一个正数
    public int firstMissingPositive(int[] nums) {
        int n = nums.length;
        for (int i = 0; i < n; i++) {
            nums[i] = nums[i] <= 0 ? n + 1 : nums[i];
        }

        for (int i = 0; i < n; i++) {
            if (Math.abs(nums[i]) <= n) {
                int ind = Math.abs(nums[i]);
                if (nums[ind - 1] > 0) {
                    nums[ind - 1] *= -1;
                }
            }
        }
        for (int i = 0; i < n; i++) {
            if (nums[i] > 0) {
                return i + 1;
            }
        }
        return n + 1;

    }

//239. 滑动窗口最大值 单调队列

    public int[] maxSlidingWindow(int[] nums, int k) {
        List<Integer> res = new ArrayList<>();
        Deque<Integer> q = new LinkedList<>();
        int l = 0, r = k - 1, n = nums.length;
        for (int i = 0; i < n; i++) {
            while (!q.isEmpty() && q.peekLast() < nums[i]) {
                q.pollLast();
            }
            q.addLast(nums[i]);

            if (i >= k - 1) {
                res.add(q.peekFirst());
                if (q.peekFirst() == nums[l++]) {
                    q.pollFirst();
                }
            }
        }
        return res.stream().mapToInt(Integer::intValue).toArray();

    }

    //134. 加油站
    public int canCompleteCircuit(int[] gas, int[] cost) {
        int sum = 0, minsum = 0, start = 0;
        for (int i = 0; i < gas.length; i++) {
            sum += (gas[i] - cost[i]);
            if (sum < minsum) {
                minsum = sum;
                start = i + 1;
            }
        }
        if (sum < 0) return -1;
        return start == gas.length ? 0 : start;
    }

    //496. 下一个更大元素 I 单调栈
    public int[] nextGreaterElement(int[] nums1, int[] nums2) {
        Map<Integer, Integer> map = new HashMap<>();
        Stack<Integer> stack = new Stack<>();
        int[] res = new int[nums2.length];
        for (int i = nums2.length - 1; i >= 0; i--) {
            map.put(nums2[i], i);
            while (!stack.empty() && stack.peek() <= nums2[i] && stack.pop() != null) ;

            res[i] = stack.isEmpty() ? -1 : stack.peek();
            stack.push(nums2[i]);
        }
        System.out.println(Arrays.stream(res).boxed().collect(Collectors.toList()));
        for (int i = 0; i < nums1.length; i++) {
            nums1[i] = res[map.get(nums1[i])];
        }
        return nums1;

    }

    //503. 下一个更大元素 II
    public int[] nextGreaterElements(int[] nums) {
//        int[] t=new int[nums.length*2];
        List<Integer> list = Arrays.stream(nums).boxed().collect(Collectors.toList());
        list.addAll(Arrays.stream(nums).boxed().collect(Collectors.toList()));
//        int[] nums2 = list.stream().mapToInt(Integer::intValue).toArray();
        Stack<Integer> stack = new Stack<>();
        int[] res = new int[list.size()];
        for (int i = list.size() - 1; i >= 0; i--) {
//            map.put(nums2[i], i);
            while (!stack.empty() && stack.peek() <= list.get(i) && stack.pop() != null) ;
            res[i] = stack.isEmpty() ? -1 : stack.peek();
            stack.push(list.get(i));
        }
        MyUtile.dis(res);
//        System.out.println(res);
        return Arrays.stream(res).boxed().collect(Collectors.toList()).subList(0, nums.length).stream().mapToInt(Integer::intValue).toArray();

    }

    //739. 每日温度
    public int[] dailyTemperatures(int[] temperatures) {
        Stack<Pair<Integer, Integer>> stack = new Stack<>();
        int[] res = new int[temperatures.length];
        for (int i = temperatures.length - 1; i >= 0; i--) {
            while (!stack.isEmpty() && stack.peek().getKey() <= temperatures[i] && stack.pop() != null) ;
            res[i] = stack.isEmpty() ? 0 : stack.peek().getValue() - i;
            stack.push(new Pair<>(temperatures[i], i));
        }
        return res;

    }

    //    //16. 最接近的三数之和
    public int threeSumClosest(int[] nums, int target) {

//    双指针解决
        Arrays.sort(nums);
        int n = nums.length, l, r, sum = 0, min = 99999;
        for (int i = 0; i < n; i++) {
            l = i + 1;
            r = n - 1;
            while (l < r) {
                sum = nums[i] + nums[l] + nums[r];
                if (sum == target) {
                    return sum;
                }
                min = Math.abs(sum - target) < Math.abs(min - target) ? sum : min;
                if (sum < target) {
                    l++;
                } else {
                    r--;
                }
            }
        }
        return min;
    }

    //
//    //27. 移除元素
    public int removeElement(int[] nums, int val) {
        if (nums == null || nums.length == 0) return 0;
        int l = 0, r = nums.length - 1, count = 0;
        while (l <= r) {
            while (l < r && nums[r] == val) {
                r--;
            }
            if (nums[l] == val) {
                swap(nums, l, r);
            }
            l++;
        }
        while (nums[r] == val) {
            r--;
        }
        return r + 1;
    }

    //    //11. 盛最多水的容器 双指针
    public int maxArea(int[] height) {
        int n = height.length, l = 0, r = n, max = 0;
        while (l < r) {
            max = Math.max((r - l) * Math.min(height[l], height[r]), max);
            if (height[l] < height[r]) {
                l++;
            } else {
                r--;
            }
        }
        return max;

    }


    public static void swap(int[] nums, int a, int b) {
        int t = nums[a];
        nums[a] = nums[b];
        nums[b] = t;
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
        int[] nums1 = new int[]{3, 2, 3};
        int[] nums2 = new int[]{3, 2, 1, 5, 6, 4};
//        numbers.myPow(2, 10);
        System.out.println(numbers.myAtoi("  -42"));

//        System.out.println(numbers.twoSum(nums1, 6));
//        System.out.println(numbers.findKthLargest3(nums1, 1));
//        System.out.println(numbers.multiply("0", "0"));
//        System.out.println(numbers.findMin3(nums1));
//        System.out.println(numbers.removeElement(nums1, 3));
//        System.out.println(numbers.findMedianSortedArrays2(nums1, nums2));
//        numbers.merge(nums1, 3, nums2, 3);
//        numbers.generate(5);
//        numbers.hammingWeight(4294967293L);
//        MyUtile.dis();
    }

}


//295. 数据流的中位数 两个优先队列
class MedianFinder {
    PriorityQueue<Integer> small;
    PriorityQueue<Integer> large;

    public MedianFinder() {
        small = new PriorityQueue<Integer>((o1, o2) -> o2 - o1);
        large = new PriorityQueue<Integer>();
    }

    public void addNum(int num) {
        if (small.size() <= large.size()) {
            large.offer(num);
            small.add(large.poll());
        } else {
            small.offer(num);
            large.add(small.poll());
        }


    }

    public double findMedian() {
        return (small.size() + large.size()) % 2 == 0 ? (small.peek() + large.peek()) / 2.0 : small.size() > large.size() ? small.peek() : large.peek();

    }
}

/**
 * Your MedianFinder object will be instantiated and called as such:
 * MedianFinder obj = new MedianFinder();
 * obj.addNum(num);
 * double param_2 = obj.findMedian();
 */


