package test3;

//数字相关题目

import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;

//import org.apache.commons.collections.CollectionUtils;

public class Numbers {
    //    179. 最大数
    public String largestNumber(int[] nums) {
        List<Long> list = Arrays.stream(nums).mapToLong(Integer::toUnsignedLong).boxed().sorted((a, b) -> (int) ((long) (b * Math.pow(10, a.toString().length()) + a) - (a * Math.pow(10, b.toString().length()) + b))).collect(Collectors.toList());
        System.out.println(list);
        StringBuilder sb = new StringBuilder();
        list.forEach(sb::append);
        while (sb.length() > 0 && sb.charAt(0) == '0') {
            sb.delete(0, 1);
        }
        return Objects.equals(sb.toString(), "") ? "0" : sb.toString();
    }

    //300. 最长递增子序列
    public int lengthOfLIS(int[] nums) {
        int n = nums.length, p = -1;
        int[] heap = new int[n];
        for (int i = 0; i < n; i++) {
            int tp = lowerBound(heap, 0, p, nums[i]);
            heap[tp] = nums[i];
            if (tp > p) {
                p++;
            }
        }
        return p + 1;
    }

    public int lowerBound(int[] heap, int l, int r, int k) {
        while (l <= r) {
            int m = l + (r - l) / 2;
            if (heap[m] == k) {
                r = m - 1;
            } else if (heap[m] < k) {
                l = m + 1;
            } else if (heap[m] > k) {
                r = m - 1;
            }
        }
        return l;
    }

    //128. 最长连续序列
    public int longestConsecutive(int[] nums) {
        int n = nums.length;
        int res = 0;
        Map<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < n; i++) {
            map.put(nums[i], 0);
        }
        for (int i = 0; i < n; i++) {
            int now = nums[i], cnt = 1;
            if (map.get(now) == 1) {
                continue;
            }
            map.put(now, 1);
            while (map.containsKey(now = now - 1)) {
                map.put(now, 1);
                cnt++;
            }
            now = nums[i];
            while (map.containsKey(now = now + 1)) {
                map.put(now, 1);
                cnt++;
            }
            res = Math.max(res, cnt);
        }
//        map.forEach((k, v) -> res.set(Math.max(res.get(), v)));
        return res;
    }


    public int hammingWeight(long n) {
        int res = 0;
        while (n != 0) {
            n &= (n - 1);
            res++;
        }
        return res;

    }

    //    22. 括号生成
    public List<String> generateParenthesis(int n) {
        List<String> res = new ArrayList<>();
        generateParenthesis(n, n, new Stack<>(), res);
        return res;
    }

    public void generateParenthesis(int l, int r, Stack<Character> stack, List<String> res) {
//        if (l < 0 || r < 0 || l > r) {
//            return;
//        }
        if (l == 0 && r == 0) {
            StringBuilder sb = new StringBuilder();
            stack.forEach(sb::append);
            res.add(sb.toString());
            return;
        }

        if (l > 0) {
            stack.push('(');
            generateParenthesis(l - 1, r, stack, res);
            stack.pop();
        }
        if (r > l) {
            stack.push(')');
            generateParenthesis(l, r - 1, stack, res);
            stack.pop();
        }

    }

    //28. 实现 strStr()
    public int strStr(String haystack, String needle) {
        int[] next = getnext(needle);
        int prenext = next[0];
        for (int i = 0; i < haystack.length(); i++) {
            while (prenext > 0 && haystack.charAt(i) != needle.charAt(prenext)) {
                prenext = next[prenext - 1];
            }
            if (haystack.charAt(i) == needle.charAt(prenext)) {
                prenext++;
            }
            if (prenext == next.length) {
                return i - next.length + 1;
            }
        }
        return -1;
    }

    public int[] getnext(String p) {
        int n = p.length();
        int[] next = new int[n];
        int prenext = next[0];
        for (int i = 1; i < n; i++) {
            while (prenext > 0 && p.charAt(i) != p.charAt(prenext)) {
                prenext = next[prenext - 1];
            }
            next[i] = p.charAt(i) == p.charAt(prenext) ? prenext = prenext + 1 : prenext;
        }
        return next;
    }

    //    1. 两数之和
    public int[] twoSum(int[] nums, int target) {
        Map<Integer, Integer> map = new HashMap<Integer, Integer>();
        for (int i = 0; i < nums.length; i++) {
            map.putIfAbsent(nums[i], i);
        }
        Arrays.sort(nums);
        int l = 0, r = nums.length - 1;
        while (l < r) {
            if (nums[l] + nums[r] == target) {
                return new int[]{map.get(nums[l]), map.get(nums[r])};
            } else if (nums[l] + nums[r] < target) {
                l++;
            } else {
                r--;
            }
        }
        return new int[]{0, 0};
    }

    //    53. 最大子数组和
    public int maxSubArray(int[] nums) {
        int n = nums.length, l = 0, r = 0, res = nums[0];
        int[] dp = new int[n];
        for (int i = 1; i < n; i++) {
//            nums[i] = Math.max(nums[i - 1] + nums[i], nums[i]);
            if (nums[i - 1] + nums[i] > nums[i]) {
                nums[i] = nums[i - 1] + nums[i];
                if (nums[i] > res) {
                    res = nums[i];
                    r = i;
                }
            } else {
                if (nums[i] > res) {
                    res = nums[i];
                }
                l = r = i;
            }
        }
//        int res = Arrays.stream(nums).max().getAsInt();
        System.out.println(l + "," + r);
        return res;

    }

//-----------2023.9.21之前：

    //    3. 无重复字符的最长子串
    public int lengthOfLongestSubstring(String s) {
        Map<Character, Integer> win = new HashMap<>();
        int l = 0, r = 0, n = s.length(), valid = 0, res = 0;
        while (r < n) {
            Character c = s.charAt(r);
            win.put(c, win.getOrDefault(c, 0) + 1);
            while (win.get(c) > 1) {
                Character tl = s.charAt(l);
                win.put(tl, win.get(tl) - 1);
                l++;
            }
            res = Math.max(res, r - l + 1);
            r++;

        }
        return res;


    }


    //438. 找到字符串中所有字母异位词
    public List<Integer> findAnagrams(String s, String p) {
        Map<Character, Integer> win = new HashMap<>();
        Map<Character, Integer> need = new HashMap<>();
        for (int i = 0; i < p.length(); i++) {
            need.put(p.charAt(i), need.getOrDefault(p.charAt(i), 0) + 1);
        }
        win = new HashMap<>(need);
        int l = 0, r = 0, n = s.length(), valid = win.size();
        List<Integer> res = new ArrayList<>();
        while (r < n) {
            Character c = s.charAt(r++);
            if (!win.containsKey(c)) {
                l = r;
                win = new HashMap<>(need);
                valid = win.size();
                continue;
            }
            win.put(c, win.get(c) - 1);
            if (win.get(c) == 0) {
                valid--;
                if (valid == 0) {
                    res.add(l);
                    Character tl = s.charAt(l);
                    win.put(tl, win.get(tl) + 1);
                    l++;
                    valid++;
                }
            }
            while (win.get(c) < 0) {
                Character tl = s.charAt(l++);
                if (win.get(tl) == 0) {
                    valid++;
                }
                win.put(tl, win.get(tl) + 1);
            }
        }
        return res;


    }

    //567. 字符串的排列
    public boolean checkInclusion(String s1, String s2) {
        Map<Character, Integer> need = new HashMap<>();
        Map<Character, Integer> win = new HashMap<>();
        for (int i = 0; i < s1.length(); i++) {
            need.put(s1.charAt(i), need.getOrDefault(s1.charAt(i), 0) + 1);
        }
        int l = 0, r = 0, n = s2.length(), valid = 0;
        while (r < n) {
            char rc = s2.charAt(r++);
            if (need.containsKey(rc)) {
                win.put(rc, win.getOrDefault(rc, 0) + 1);
                if (Objects.equals(win.get(rc), need.get(rc))) {
                    valid++;
                    if (valid == need.size()) {
                        return true;
                    }
                }

                while (win.get(rc) > need.get(rc)) {
                    char lc = s2.charAt(l++);
                    if (Objects.equals(win.get(lc), need.get(lc))) {
                        valid--;
                    }
                    win.put(lc, win.get(lc) - 1);
                }


            } else {
                l = r;
                win.clear();
                valid = 0;
            }
        }
        return false;


    }

    //76. 最小覆盖子串
    public String minWindow(String s, String t) {
        Map<Character, Integer> need = new HashMap<>();
        Map<Character, Integer> win = new HashMap<>();
        for (int i = 0; i < t.length(); i++) {
            need.put(t.charAt(i), need.getOrDefault(t.charAt(i), 0) + 1);
        }
        int l = 0, r = 0, n = s.length(), valid = 0, res = 999999, resl = 0, resr = 0;
        while (r < n) {
            char rc = s.charAt(r++);

            if (need.containsKey(rc)) {
                win.put(rc, win.getOrDefault(rc, 0) + 1);
                if (Objects.equals(win.get(rc), need.get(rc))) {
                    valid++;
                }
                while (valid == need.size()) {
                    if (res > r - l) {
                        res = r - l;
                        resl = l;
                        resr = r;
                    }
                    char lc = s.charAt(l++);
                    if (need.containsKey(lc)) {
                        if (Objects.equals(win.get(lc), need.get(lc))) {
                            valid--;
                        }
                        win.put(lc, win.get(lc) - 1);
                    }
                }
            }
//            System.out.println(s.substring(resl, resr));
        }
        return s.substring(resl, resr);


    }

    public static void main(String[] args) {
        Numbers nu = new Numbers();
//        System.out.println(nu.findAnagrams("vwwvv", "vwv"));
        System.out.println(nu.hammingWeight(42949672933L));

    }
}





