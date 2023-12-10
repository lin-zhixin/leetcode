package test2;

import javafx.util.Pair;

import java.util.*;

// 字符串处理
public class Strings {

    //    3. 无重复字符的最长子串
    public int lengthOfLongestSubstring(String s) {
        Map<Character, Integer> win = new HashMap<>();
        int l = 0, r = 0, maxlen = -9999;
        while (r < s.length()) {
            char cr = s.charAt(r++);
            win.put(cr, win.getOrDefault(cr, 0) + 1);
            while (win.get(cr) > 1) {
                char cl = s.charAt(l++);
                win.put(cl, win.get(cl) - 1);
            }
            maxlen = Math.max(maxlen, r - l);
        }
        System.out.println(maxlen);
        return maxlen == -9999 ? 0 : maxlen;
    }

    //
//    //    49. 字母异位词分组
    public List<List<String>> groupAnagrams(String[] strs) {
//       每个字符串排序确定唯一性 之后再放入map
        Map<String, List<String>> res = new HashMap<String, List<String>>();

        for (int i = 0; i < strs.length; i++) {
            char[] t = strs[i].toCharArray();
            Arrays.sort(t);
            String key = new String(t);
            Arrays.toString(t);
            List<String> tlist = res.getOrDefault(key, new ArrayList<>());
            tlist.add(strs[i]);
            res.put(key, tlist);
        }
        System.out.println(new ArrayList<>(res.values()));
        return new ArrayList<>(res.values());
    }

    //
//
//    //    KMP算法   28. 找出字符串中第一个匹配项的下标
    public int strStr(String haystack, String needle) {
        int[] next = getNext(needle);

        for (int i = 0, k = 0; i < haystack.length(); i++) {
            while (k > 0 && haystack.charAt(i) != needle.charAt(k)) {
                k = next[k - 1];

            }
            if (haystack.charAt(i) == needle.charAt(k)) {
                k++;
            }
            if (k == needle.length()) {
                return i - k + 1;
            }

        }
        return -1;
    }

    public int[] getNext(String p) {
        int n = p.length(), k = 0;
        int[] next = new int[n];//存放的是长度 不是下标 只是在下面的while里面当做下标来用
        next[0] = 0;
        for (int i = 1; i < n; i++) {
//            k就是上一个的next 就是next[i-1]
            while (k > 0 && p.charAt(i) != p.charAt(k)) {
                k = next[k - 1];//减一是因为这边k做下标用，和长度差1
            }
            if (p.charAt(i) == p.charAt(k)) {
                k++;//减一是因为这边k做下标用，和长度差1
            }
            next[i] = k;
        }
        return next;
    }

    //
//    //151. 反转字符串中的单词 先整体反转再局部反转
    public String reverseWords(String s) {
        s = reverse(new StringBuilder(s)).trim();
        String[] strings = s.split(" ");
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < strings.length; i++) {
            if (!"".equals(strings[i])) {
                sb.append(reverse(new StringBuilder(strings[i])));
                if (i != strings.length - 1) {
                    sb.append(" ");
                }
            }
        }
        return sb.toString();
    }

    public String reverse(StringBuilder s) {
        char[] c = s.toString().toCharArray();
        int l = 0, r = s.length() - 1;
        while (l < r) {
            char t = c[l];
            c[l] = c[r];
            c[r] = t;
            l++;
            r--;
        }

//        System.out.println(Arrays.toString(c));
        return String.valueOf(c);
    }

    //
//
//    //    395. 至少有 K 个重复字符的最长子串
    public int longestSubstring(String s, int k) {
        int max = 0;
//        因为取的是最大值，所以只能通过枚举所有的可能性，就是窗口内所有可能的字母数量 之后取最大
        for (int i = 1; i <= 26; i++) {
            int lessKNum = 0, types = 0, l = 0, r = 0, n = s.length();
            HashMap<Character, Integer> win = new HashMap<>();
            while (r < n) {
                char cr = s.charAt(r++);
                win.put(cr, win.getOrDefault(cr, 0) + 1);
                if (win.get(cr) == 1) {
                    lessKNum++;
                    types++;
                }
                if (win.get(cr) == k) {
                    lessKNum--;
                }

//                窗口偶收缩时机：字母类型数量超过枚举的i值的时候收缩，否则越长越好。
                while (types > i) {
                    char cl = s.charAt(l++);
                    win.put(cl, win.get(cl) - 1);
                    if (win.get(cl) == k - 1) {
                        lessKNum++;
                    }
                    if (win.get(cl) == 0) {
                        types--;
                        lessKNum--;
                    }
                }
                if (lessKNum == 0) {
//                    System.out.println(s.substring(l,r));
                    max = Math.max(max, r - l);
                }
            }
        }
        return max;
    }

    //
//    //13. 罗马数字转整数
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

    //
////14. 最长公共前缀
//
    public String longestCommonPrefix(String[] strs) {
        if (strs == null || strs.length == 0) {
            return "";
        }
        int l = 0, r = Arrays.stream(strs).mapToInt(String::length).min().orElse(0);
        while (l < r) {
//            [)
            int mid = l + (r - l + 1) / 2;
            if (isCommonPrefix(strs, mid)) {
                l = mid;
            } else {
                r = mid - 1;
            }
        }
        return strs[0].substring(0, l);
    }

    public boolean isCommonPrefix(String[] strs, int len) {
        return Arrays.stream(strs)
                .allMatch(s -> s.substring(0, len).equals(strs[0].substring(0, len)));
    }

    //
//    //125. 验证回文串
    public boolean isPalindrome(String s) {
        s = s.toLowerCase();
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if (c >= 'a' && c <= 'z' || c >= '0' && c <= '9') {
                sb.append(c);
            }
        }
        int l = 0, r = sb.length() - 1;
        while (l < r) {
            if (sb.charAt(l++) != sb.charAt(r--)) {
                return false;
            }
        }
        return true;
    }

    //
//    //171. Excel 表列序号
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

    //
//    //242. 有效的字母异位词
    public boolean isAnagram(String s, String t) {
        Map<Character, Integer> map = new HashMap<>();

        for (int i = 0; i < s.length(); i++) {
            map.put(s.charAt(i), map.getOrDefault(s.charAt(i), 0) + 1);
        }
        for (int i = 0; i < t.length(); i++) {
            char c = t.charAt(i);
            if (!map.containsKey(c)) {
                return false;
            }
            map.compute(c, (k, v) -> v - 1);
        }
//        return Optional.ofNullable(map).map()
        return map.values().stream().allMatch(v -> v == 0);


//        for (int i = 0; i < t.length(); i++) {
//            if (!map.containsKey(t.charAt(i)) || map.get(t.charAt(i)) == 0) {
//                return false;
//            } else {
////                map.put(t.charAt(i), map.get(t.charAt(i)) - 1);
//                map.compute(t.charAt(i), (k, v) -> v - 1);
//            }
//        }
//        return map.values().stream().allMatch(m -> m == 0);

    }

    //
//    //344. 反转字符串
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

    //
//    //387. 字符串中的第一个唯一字符
    public int firstUniqChar(String s) {
        Deque<Pair<Character, Integer>> q = new LinkedList<>();
        Map<Character, Integer> map = new HashMap<>();

        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if (!map.containsKey(c) || map.get(c) != -1) {
                map.put(c, map.getOrDefault(c, 0) + 1);
                q.offerLast(new Pair<>(c, i));
            }
            if (map.get(c) > 1) {
                map.put(c, -1);
            }
            while (!q.isEmpty() && map.get(q.peekFirst().getKey()) == -1) {
                q.pollFirst();
            }
        }
        return Objects.isNull(q.peek()) ? -1 : q.peek().getValue();

    }

    //
//    //412. Fizz Buzz
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

    //
//    //316. 去除重复字母 单调栈
    public String removeDuplicateLetters(String s) {
//       存放去重的结果 里面没有重复的
        Stack<Character> stack = new Stack<>();
//        栈中是否存在
        Map<Character, Boolean> instack = new HashMap<>();
//        剩余字符串的各个字符数量
        Map<Character, Integer> count = new HashMap<>();

        int n = s.length();
        for (int i = 0; i < n; i++) {
            count.put(s.charAt(i), count.getOrDefault(s.charAt(i), 0) + 1);
        }

        for (int i = 0; i < n; i++) {
            char c = s.charAt(i);
            count.compute(c, (k, v) -> v - 1);
            if (Objects.equals(instack.get(c), true)) {
                continue;
            }

            while (!stack.isEmpty() && stack.peek() > c && count.get(stack.peek()) > 0) {
                instack.put(stack.pop(), false);
            }
            stack.push(c);
            instack.put(c, true);
        }
        StringBuilder sb=new StringBuilder();
        stack.forEach(sb::append);
        return sb.toString();

    }

    //
//
    public static void main(String[] args) {
        Strings obj = new Strings();
//        System.out.printf("" + obj.groupAnagrams(new String[]{"eat", "tea", "tan", "ate", "nat", "bat"}));
//        System.out.println(obj.reverseWords("the    sky   is   blue"));
//        System.out.println(obj.longestCommonPrefix(new String[]{"flower", "low", "flight"}));
        System.out.println(obj.removeDuplicateLetters("cbacdcbc"));
//        System.out.println(obj.isAnagram("ab", "a"));

    }
}
