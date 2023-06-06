package category;

import javafx.util.Pair;

import java.util.*;

// 字符串处理
public class Strings {

    //    3. 无重复字符的最长子串
    public int lengthOfLongestSubstring(String s) {
        if (Objects.isNull(s) || s.length() == 0) return 0;
        Map<Character, Integer> map = new HashMap<>();
        Integer left = 0;
        Integer max = 0;
        for (int i = 0; i < s.length(); i++) {
            if (map.containsKey(s.charAt(i))) {
//                left=map.get(s.charAt(i))+1;
                left = Math.max(left, map.get(s.charAt(i)) + 1);//left要取最大值

            }
            map.put(s.charAt(i), i);
            max = Math.max(max, i - left + 1);
        }
        return max;
    }

    //    49. 字母异位词分组
    public List<List<String>> groupAnagrams(String[] strs) {
//       每个字符串排序确定唯一性 之后再放入map
        Map<String, List<String>> map = new HashMap<>();
        for (String s : strs) {
            char[] chars = s.toCharArray();
            Arrays.sort(chars);
            String key = new String(chars);
            List<String> tempList = map.getOrDefault(key, new ArrayList<String>());
            tempList.add(s);
            map.put(key, tempList);
        }
        return new ArrayList<List<String>>(map.values());
    }


    //    KMP算法   28. 找出字符串中第一个匹配项的下标
    public int strStr(String haystack, String needle) {
        int[] next = getNext(needle);
        for (int i = 0, j = 0; i < haystack.length(); i++) {
            while (j > 0 && haystack.charAt(i) != needle.charAt(j)) {
                j = next[j - 1];
            }
            if (haystack.charAt(i) == needle.charAt(j)) {
                j++;
            }
            if (j == needle.length()) {
                return i - j + 1;
            }
        }
        return -1;
    }

    public int[] getNext(String p) {
        int[] next = new int[p.length()];
        next[0] = 0;
        for (int i = 1, j = 0; i < p.length(); i++) {
            while (j > 0 && p.charAt(i) != p.charAt(j)) {
                j = next[j - 1];
            }
            if (p.charAt(i) == p.charAt(j)) {
                j++;
            }
            next[i] = j;
        }
        return next;
    }

    //151. 反转字符串中的单词 先整体反转再局部反转
    public String reverseWords(String s) {
        StringBuilder ss = new StringBuilder(reverse(new StringBuilder(s)));
        int l = 0, r = 0, pre = -1;
        while (r <= ss.length()) {
            if (r == ss.length() || Objects.equals(ss.charAt(r), ' ')) {
                if (r > 0 && r < ss.length() && (Objects.equals(ss.charAt(r - 1), ' ') || Objects.equals(ss.charAt(r), '\0'))) {
                    ss.replace(r, r + 1, "");
                } else {
                    ss.replace(l, r, reverse(new StringBuilder(ss.substring(l, r))));
                    l = ++r;
                }
            } else {
                r++;
            }
        }
        return ss.toString().trim();
    }

    public String reverse(StringBuilder s) {
        int l = 0, r = s.length() - 1;
        while (l < r) {
            char c = s.charAt(l);
            s.setCharAt(l, s.charAt(r));
            s.setCharAt(r, c);
            l++;
            r--;
        }
        return s.toString();
    }


    //    395. 至少有 K 个重复字符的最长子串
    public int longestSubstring(String s, int k) {
        int max = 0, n = s.length();
        for (int i = 1; i <= 26; i++) {
            int l = 0, r = 0, types = 0, lessKNum = 0;
            int[] cnt = new int[26];
            while (r < n) {
                int rnum = s.charAt(r) - 'a';
                cnt[rnum]++;
                if (cnt[rnum] == 1) {
                    types++;
                    lessKNum++;
                }
                if (cnt[rnum] == k) {
                    lessKNum--;
                }


                while (types > i) {
                    int lnum = s.charAt(l) - 'a';
                    cnt[lnum]--;
                    if (cnt[lnum] == 0) {
                        types--;
                        lessKNum--;
                    }
                    if (cnt[lnum] == k - 1) {
                        lessKNum++;
                    }
                    l++;
                }
                if (lessKNum == 0) {
                    max = Math.max(max, r - l + 1);
                }
                r++;
            }

        }
        return max;

    }

    //13. 罗马数字转整数
    public int romanToInt(String s) {
        Map<Character, Integer> map = new HashMap<>();
        map.put('I', 1);
        map.put('V', 5);
        map.put('X', 10);
        map.put('L', 50);
        map.put('C', 100);
        map.put('D', 500);
        map.put('M', 1000);
//        map.put("IV",4);
//        map.put("IV",4);
//        map.put("XL",1000);
//        map.put("XC",1000);
//        map.put("CD",1000);
//        map.put("CM",1000);
        int res = 0;
        for (int i = 0; i < s.length(); i++) {
            if (s.charAt(i) == 'I' && (i + 1 < s.length()) && (s.charAt(i + 1) == 'V' || s.charAt(i + 1) == 'X')) {
                res -= map.get(s.charAt(i));
            } else if (s.charAt(i) == 'X' && (i + 1 < s.length()) && (s.charAt(i + 1) == 'L' || s.charAt(i + 1) == 'C')) {
                res -= map.get(s.charAt(i));
            } else if (s.charAt(i) == 'C' && (i + 1 < s.length()) && (s.charAt(i + 1) == 'D' || s.charAt(i + 1) == 'M')) {
                res -= map.get(s.charAt(i));
            } else {
                res += map.get(s.charAt(i));
            }

        }
        return res;

    }

//14. 最长公共前缀

    public String longestCommonPrefix(String[] strs) {
        if (strs == null || strs.length == 0) {
            return "";
        }
        if (strs.length == 1) return strs[0];

        int len = 9999;
        for (int i = 0; i < strs.length; i++) {
            len = Math.min(strs[i].length(), len);
        }
        int l = 0, r = len;
        while (l < r) {
            int mid = l + (r - l + 1) / 2;
            if (isCommonPrefix(strs, mid)) {
                l = mid;
            } else {
                r = mid - 1;
            }
        }
        System.out.println(strs[0].substring(0, l));
        return strs[0].substring(0, l);
    }

    public boolean isCommonPrefix(String[] strs, int len) {
//        if (len==0)return false;
        String str0 = strs[0].substring(0, len);
        for (int i = 1; i < strs.length; i++) {
            if (!str0.equals(strs[i].substring(0, len))) {
                return false;
            }
        }
        return true;
    }

    //125. 验证回文串
    public boolean isPalindrome(String s) {
        s = s.toLowerCase();
        StringBuilder ss = new StringBuilder();
        for (int i = 0; i < s.length(); i++) {
            if (s.charAt(i) >= 'A' && s.charAt(i) <= 'Z' || s.charAt(i) >= 'a' && s.charAt(i) <= 'z' || s.charAt(i) >= '0' && s.charAt(i) <= '9') {
                ss.append(s.charAt(i));
            }
        }
        System.out.println(ss);
        int l = 0, r = ss.length() - 1;
        while (l < r) {
            if (ss.charAt(l++) != ss.charAt(r--)) {
                return false;
            }
        }
        return true;

    }

    //171. Excel 表列序号
    public int titleToNumber(String columnTitle) {
        Map<Character, Integer> map = new HashMap<>();
        for (char i = 'A'; i <= 'Z'; i++) {
            map.put(i, i - 'A' + 1);
        }

        int p = 0;
        int res = 0;
        for (int i = columnTitle.length() - 1; i >= 0; i--) {
            res += map.get(columnTitle.charAt(i)) * Math.pow(26, p);
            p++;
        }
        System.out.println(res);
        return res;

    }

    //242. 有效的字母异位词
    public boolean isAnagram(String s, String t) {
        Map<Character, Integer> map = new HashMap<>();
        for (int i = 0; i < s.length(); i++) {
            map.put(s.charAt(i), map.getOrDefault(s.charAt(i), 0) + 1);
        }
        for (int i = 0; i < t.length(); i++) {
            if (!map.containsKey(t.charAt(i)) || map.get(t.charAt(i)) == 0) {
                return false;
            } else {
//                map.put(t.charAt(i), map.get(t.charAt(i)) - 1);
                map.compute(t.charAt(i), (k, v) -> v - 1);
            }
        }
        return map.values().stream().allMatch(m -> m == 0);

    }

    //344. 反转字符串
    public void reverseString(char[] s) {
        int l = 0, r = s.length - 1;
        while (l < r) {
            if (s[l] != s[r]) {
                char t = s[l];
                s[l] = s[r];
                s[r] = t;
            }
            l++;
            r--;
        }

    }

    //387. 字符串中的第一个唯一字符
    public int firstUniqChar(String s) {
        Deque<Pair<Character, Integer>> queue = new LinkedList<>();
//        Deque<Character> queue = new LinkedList<>();
        Map<Character, Integer> map = new HashMap<>();
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if (!map.containsKey(c)) {
                map.put(c, 1);
                queue.offer(new Pair<>(c, i));
            } else {
                map.compute(c, (k, v) -> -1);
                while (!queue.isEmpty() && map.get(queue.peek().getKey()) == -1) {
                    queue.poll();
                }

            }
        }
        return queue.isEmpty() ? -1 : queue.peek().getValue();


    }

    //412. Fizz Buzz
    public List<String> fizzBuzz(int n) {
        List<String> res = new ArrayList<>();
        for (int i = 1; i <= n; i++) {
            if ((i % 3) == 0 && (i % 5) == 0) {
                res.add("FizzBuzz");
            } else if ((i % 3) == 0) {
                res.add("Fizz");
            } else if ((i % 5) == 0) {
                res.add("Buzz");
            } else {
                res.add(String.valueOf(i));
            }

        }
        return res;

    }





    public static void main(String[] args) {
        Strings obj = new Strings();
//        System.out.printf("" + obj.groupAnagrams(new String[]{"eat", "tea", "tan", "ate", "nat", "bat"}));
//        System.out.println(obj.reverseWords("the    sky   is   blue"));
//        System.out.println(obj.longestCommonPrefix(new String[]{"flower", "low", "flight"}));
//        System.out.println(obj.isPalindrome("ab_a"));
//        obj.reverseBits(4294967293);

    }
}
