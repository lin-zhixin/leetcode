package test3;

import java.util.*;
import java.util.stream.IntStream;

public class DFS {
    //    46. 全排列
    public List<List<Integer>> permute(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        permute(nums, 0, new boolean[nums.length], new ArrayDeque<>(), res);
        System.out.println(res);
        return res;
    }

    public void permute(int[] nums, int i, boolean[] v, Deque<Integer> q, List<List<Integer>> res) {
        int n = nums.length;
        if (q.size() == n) {
            res.add(new ArrayList<>(q));
            return;
        }
//        if (v[i]) {
//            return;
//        }
        for (int j = 0; j < n; j++) {
            if (!v[j]) {
                v[j] = true;
                q.offerLast(nums[j]);
                permute(nums, i, v, q, res);
                q.pollLast();
                v[j] = false;
            }
        }
    }

    //47. 全排列 II
    public List<List<Integer>> permuteUnique(int[] nums) {
        Arrays.sort(nums);
        List<List<Integer>> res = new ArrayList<>();
        permuteUnique(nums, 0, new boolean[nums.length], new ArrayDeque<>(), res);
        System.out.println(res);
        return res;
    }

    public void permuteUnique(int[] nums, int i, boolean[] v, Deque<Integer> q, List<List<Integer>> res) {
        int n = nums.length;
        if (q.size() == n) {
            res.add(new ArrayList<>(q));
            return;
        }
        int pre = Integer.MAX_VALUE;
        for (int j = 0; j < n; j++) {
            if (!v[j] && pre != nums[j]) {
                v[j] = true;
                q.offerLast(pre = nums[j]);
                permuteUnique(nums, i, v, q, res);
                q.pollLast();
                v[j] = false;
            }
        }
    }

    //77. 组合
    public List<List<Integer>> combine(int n, int k) {
        int[] nums = IntStream.range(1, n + 1).toArray();
        List<List<Integer>> res = new ArrayList<>();
        combine(nums, 0, k, new boolean[nums.length], new ArrayDeque<>(), res);
        System.out.println(res);
        return res;

    }

    public void combine(int[] nums, int i, int k, boolean[] v, Deque<Integer> q, List<List<Integer>> res) {
        int n = nums.length;
        if (q.size() == k) {
            res.add(new ArrayList<>(q));
            return;
        }
        int pre = Integer.MAX_VALUE;
        for (int j = i; j < n; j++) {
            if (!v[j] && pre != nums[j]) {
                v[j] = true;
                q.offerLast(pre = nums[j]);
                combine(nums, j, k, v, q, res);
                q.pollLast();
                v[j] = false;
            }
        }
    }

    //39. 组合总和
    public List<List<Integer>> combinationSum(int[] candidates, int target) {
        Arrays.sort(candidates);
        List<List<Integer>> res = new ArrayList<>();
        combinationSum(candidates, 0, target, new boolean[candidates.length], new ArrayDeque<>(), res);
        System.out.println(res);
        return res;


    }

    public void combinationSum(int[] nums, int i, int k, boolean[] v, Deque<Integer> q, List<List<Integer>> res) {
        int n = nums.length;
        int sum = q.stream().mapToInt(Integer::intValue).sum();
        if (sum == k) {
            res.add(new ArrayList<>(q));
            return;
        }
//        if (sum > k) {
//            return;
//        }
        for (int j = i; j < n && sum < k; j++) {
            q.offerLast(nums[j]);
            combinationSum(nums, j, k, v, q, res);
            q.pollLast();
        }
    }

    //40. 组合总和 II
    public List<List<Integer>> combinationSum2(int[] candidates, int target) {
        Arrays.sort(candidates);
        List<List<Integer>> res = new ArrayList<>();
        combinationSum2(candidates, 0, target, new boolean[candidates.length], new ArrayDeque<>(), res);
        System.out.println(res);
        return res;


    }

    public void combinationSum2(int[] nums, int i, int k, boolean[] v, Deque<Integer> q, List<List<Integer>> res) {
        int n = nums.length;
        int sum = q.stream().mapToInt(Integer::intValue).sum();
        if (sum == k) {
            res.add(new ArrayList<>(q));
            return;
        }
//        if (sum > k) {
//            return;
//        }
        int pre = Integer.MAX_VALUE;
        for (int j = i; j < n && sum < k; j++) {
//
            if (pre != nums[j]) {
                q.offerLast(pre = nums[j]);
                combinationSum2(nums, j + 1, k, v, q, res);
                q.pollLast();
            }
        }
    }

    //216. 组合总和 III
    public List<List<Integer>> combinationSum3(int k, int n) {
        int[] nums = IntStream.range(1, 10).toArray();
        List<List<Integer>> res = new ArrayList<>();
        combinationSum3(nums, 0, n, k, new boolean[nums.length], new ArrayDeque<>(), res);
        System.out.println(res);
        return res;

    }

    public void combinationSum3(int[] nums, int i, int k, int len, boolean[] v, Deque<Integer> q, List<List<Integer>> res) {
        int n = nums.length;
        int sum = q.stream().mapToInt(Integer::intValue).sum();
        if (q.size() == len && sum == k) {
            res.add(new ArrayList<>(q));
            return;
        }
        int pre = Integer.MAX_VALUE;
        for (int j = i; j < n && sum < k; j++) {
            if (pre != nums[j]) {
                q.offerLast(pre = nums[j]);
                combinationSum3(nums, j + 1, k, len, v, q, res);
                q.pollLast();
            }
        }
    }

    //78. 子集 90. 子集 II
    public List<List<Integer>> subsets(int[] nums) {
        Arrays.sort(nums);
        List<List<Integer>> res = new ArrayList<>();
        subsets(nums, 0, new ArrayDeque<>(), res);
        System.out.println(res);
        return res;
    }

    public void subsets(int[] nums, int i, Deque<Integer> q, List<List<Integer>> res) {
        int n = nums.length;
        res.add(new ArrayList<>(q));
        int pre = Integer.MAX_VALUE;
        for (int j = i; j < n; j++) {
            if (pre != nums[j]) {
                q.offerLast(pre = nums[j]);
                subsets(nums, j + 1, q, res);
                q.pollLast();
            }
        }
    }


}
