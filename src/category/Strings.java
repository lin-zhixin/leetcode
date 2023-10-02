package category;

import javafx.util.Pair;

import java.util.*;

// 字符串处理
public class Strings {
    //    394. 字符串解码
    public String decodeString(String s) {
        Deque<Character> stack = new LinkedList<>();
        int n = s.length();
//        boolean
        StringBuilder res = new StringBuilder();
        for (int i = 0; i < n; i++) {
            char c = s.charAt(i);
            if (c != ']') {
                stack.offerLast(c);
            } else {
                StringBuilder t = new StringBuilder(), tres = new StringBuilder(), numsb = new StringBuilder();
                while (!stack.isEmpty() && stack.peekLast() != '[') {
                    t.append(stack.pollLast());
                }
                stack.pollLast();
                t.reverse();
                while (!stack.isEmpty() && Character.isDigit(stack.peekLast())) {
                    numsb.append(stack.pollLast());
                }
                numsb.reverse();
                int num = Integer.parseInt(numsb.toString());
                while ((num = num - 1) >= 0) {
                    tres.append(t);
                }
                for (int j = 0; j < tres.length(); j++) {
                    stack.offerLast(tres.charAt(j));
                }
            }
        }
        stack.forEach(res::append);
//        System.out.println(res);
        return res.toString();

    }


    //    ----------------------------2023.9.21之前：
    //    392. 判断子序列 二分查找
    public boolean isSubsequence(String s, String t) {
        Map<Character, List<Integer>> map = new HashMap<>();
        for (int i = 0; i < t.length(); i++) {
            char c = t.charAt(i);
            int finalI = i;
            map.compute(c, (k, v) -> {
                if (Objects.isNull(v)) {
                    v = new ArrayList<>();
                }
                v.add(finalI);
                return v;
            });
        }
        System.out.println(map);
        int j = 0;
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if (!map.containsKey(c)) {
                return false;
            }
            int pos = lower_bound(map.get(c), j);
            if (pos == map.get(c).size()) {
                return false;
            }
            j = map.get(c).get(pos) + 1;
        }
        return true;

    }

    public int lower_bound(List<Integer> list, int k) {
        int l = 0, r = list.size(), mid;
        while (l < r) {
//            [)
            mid = l + (r - l) / 2;
            if (list.get(mid) == k) {
                r = mid;
            } else if (list.get(mid) < k) {
                l = mid + 1;
            } else if (list.get(mid) > k) {
                r = mid;
            }
        }
        return l;
    }

    //    792. 匹配子序列的单词数
    public int numMatchingSubseq(String s, String[] words) {
        Map<Character, List<Integer>> map = new HashMap<>();
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            int finalI = i;
            map.compute(c, (k, v) -> {
                if (Objects.isNull(v)) {
                    v = new ArrayList<>();
                }
                v.add(finalI);
                return v;
            });
        }
        System.out.println(map);
        int sum = 0;
        for (String word : words) {
            int j = 0;
            for (int i = 0; i < word.length(); i++) {
                char c = word.charAt(i);
                if (!map.containsKey(c)) {
                    sum--;
                    break;
                }
                int pos = lower_bound(map.get(c), j);
                if (pos == map.get(c).size()) {
                    sum--;
                    break;
                }
                j = map.get(c).get(pos) + 1;
            }
            sum++;

        }
        return sum;

    }

    //    20. 有效的括号
    public boolean isValid(String s) {
        Stack<Character> stack = new Stack<>();
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if (c == '(' || c == '[' || c == '{') {
                stack.push(c);
            } else if (!stack.isEmpty()) {
                if (c == ')' && stack.peek() == '(' || c == ']' && stack.peek() == '[' || c == '}' && stack.peek() == '{') {
                    stack.pop();
                } else {
                    return false;
                }
            } else {
                return false;
            }
        }
        return stack.isEmpty();
    }

    //    921. 使括号有效的最少添加
    public int minAddToMakeValid(String s) {
        int rightNeed = 0, res = 0;
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if (c == '(') {
                rightNeed++;
            } else {
                rightNeed--;
            }
            if (rightNeed == -1) {
                rightNeed = 0;
                res++;
            }
        }
        return res + rightNeed;

    }

    //1541. 平衡括号字符串的最少插入次数
    public int minInsertions(String s) {
//        "(((()(()((())))(((()())))()())))(((()(()()((()()))"   31
//        "(()))(()))()())))"   4   ()
        int sum = 0, rightNeed = 0, res = 0;
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if (c == '(') {
                if (rightNeed % 2 == 1) {
//                    当前是左括号，如果前面的右括号需求如果是奇数说明至少需要再插入一个右括号
//                    因为奇数个右括号必然不能匹配，如：((((()(  rightNeed是9，此时遍历到最后一个，是（，因此前面必须要插入至少一个右括号
                    res++;
                    rightNeed--;
                }
                rightNeed += 2;
            } else {
                rightNeed--;
                if (rightNeed == -1) {
//                    说明多了一个右括号，此时左边已经遍历过了 后面也是没法改变前面的情况的，因此只能插入一个左括号
//                    因为插入一个左括号导致右括号的需求增加了一个
                    res++;
                    rightNeed = 1;
                }
            }
        }
        return res + rightNeed;

    }

    //    241. 为运算表达式设计优先级 分治法
    public List<Integer> diffWaysToCompute(String expression) {
        List<Integer> res = new ArrayList<>();
        int n = expression.length();
        if (n == 0) {
            return res;
        }
        for (int i = 0; i < n; i++) {
            char c = expression.charAt(i);
            if (c == '-' || c == '*' || c == '+') {
                List<Integer> left = diffWaysToCompute(expression.substring(0, i));
                List<Integer> right = diffWaysToCompute(expression.substring(i + 1, n));
                left.forEach(l ->
                        right.forEach(r -> {
                            if (c == '-') res.add(l - r);
                            if (c == '*') res.add(l * r);
                            if (c == '+') res.add(l + r);
                        })
                );
            }
        }
        if (res.isEmpty()) {
            res.add(Integer.parseInt(expression));
        }
        return res;
    }

    //76. 最小覆盖子串
    public String minWindow(String s, String t) {
        Map<Character, Integer> need = new HashMap<>();
        Map<Character, Integer> window = new HashMap<>();
        for (int i = 0; i < t.length(); i++) {
            need.put(t.charAt(i), need.getOrDefault(t.charAt(i), 0) + 1);
        }
        int l = 0, r = 0, start = 0, valid = 0, minlen = 99999;
        while (r < s.length()) {
            char cr = s.charAt(r++);
            if (need.containsKey(cr)) {
                window.put(cr, window.getOrDefault(cr, 0) + 1);
                if (Objects.equals(need.get(cr), window.get(cr))) {
                    valid++;
                }

            }
//            r++;
            while (valid == need.size()) {
                if (r - l < minlen) {
                    start = l;
                    minlen = r - l;
                }
                char cl = s.charAt(l++);
                if (need.containsKey(cl)) {
                    if (Objects.equals(need.get(cl), window.get(cl))) {
                        valid--;
                    }
                    window.put(cl, window.get(cl) - 1);
//                    l++;
                }
            }
//            System.out.println(s.substring(start, start + minlen));

        }
        System.out.println(s.substring(start, start + minlen));
        return minlen == 99999 ? "" : s.substring(start, start + minlen);

    }

    //    567. 字符串的排列
    public boolean checkInclusion(String s1, String s2) {
        Map<Character, Integer> need = new HashMap<>();
        Map<Character, Integer> win = new HashMap<>();
        for (int i = 0; i < s1.length(); i++) {
            need.put(s1.charAt(i), need.getOrDefault(s1.charAt(i), 0) + 1);
        }

        int l = 0, r = 0, valid = 0;
        while (r < s2.length()) {
            char cr = s2.charAt(r++);
            if (need.containsKey(cr)) {
                win.put(cr, win.getOrDefault(cr, 0) + 1);
                if (Objects.equals(need.get(cr), win.get(cr))) {
                    valid++;
                }
                while (win.get(cr) > need.get(cr)) {
                    char cl = s2.charAt(l++);
                    if (Objects.equals(need.get(cl), win.get(cl))) {
                        valid--;
                    }
                    win.put(cl, win.get(cl) - 1);
                }
                if (valid == need.size()) {
                    return true;
                }
            } else {
                l = r;
                win.clear();
                valid = 0;
            }

        }
        return false;


    }

    //438. 找到字符串中所有字母异位词
    public List<Integer> findAnagrams(String s, String p) {
        Map<Character, Integer> need = new HashMap<>();
        Map<Character, Integer> win = new HashMap<>();
        List<Integer> res = new ArrayList<>();
        for (int i = 0; i < p.length(); i++) {
            need.put(p.charAt(i), need.getOrDefault(p.charAt(i), 0) + 1);
        }
        int l = 0, r = 0, valid = 0;
        while (r < s.length()) {
            char cr = s.charAt(r++);
            if (need.containsKey(cr)) {
                win.put(cr, win.getOrDefault(cr, 0) + 1);
                if (Objects.equals(win.get(cr), need.get(cr))) {
                    valid++;
                }
//                存在刚加的字符数量大于need的数量，因此需要缩小窗口到刚好等于need的窗口
                while (win.get(cr) > need.get(cr)) {
                    char cl = s.charAt(l++);
                    if (Objects.equals(win.get(cl), need.get(cl))) {
                        valid--;
                    }
                    win.put(cl, win.get(cl) - 1);
                }
//                满足条件 加入结果集并且缩小窗口
                if (valid == need.size()) {
                    res.add(l);
                    char cl = s.charAt(l++);
                    win.put(cl, win.get(cl) - 1);
                    valid--;
                }
            } else {
//                如果出现一个不在里面的说明这个字符串肯定不是 因此从下一个重新来检测
                l = r;
                win.clear();
                valid = 0;
            }
        }
        System.out.println(res);
        return res;

    }

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
        int n = p.length();
        //next数组里面存放的是最大前后缀的下一位位置
//        其求值过程就是（比如next[i]）：i位置与[0~i-1]的最大前后缀的下一位（其实就是next[i-1]）比较是否相等,
//        如果相等那就说明[0~i]这个字符串的最大前后缀可以加一位了，比如abckdjsabck ,
//        在比较最后一位k的时候去找到abckdjsabc的最大前后缀（是abc）的下一位进行比较，如果相等那就说明整个字符串的最大前后缀可以加1了，
//        也就是整个字符串的最大前后缀变成abck，如果不相等就再查找abc的最大前后缀的下一位作比较，以此类推。直到最终比较到第0位还不相等那就是0
        int[] next = new int[n];
        next[0] = 0;
        int k = next[0];
        for (int i = 1; i < n; i++) {
//            每次循环开始 k就是上一个的next 就是next[i-1]：前面子串的最大前后缀的下一位字符p[]
            while (k > 0 && p.charAt(i) != p.charAt(k)) {
                k = next[k - 1];//减一是因为这边k做下标用，和最大前后缀的下一个位置差1
            }
//            如果i位置等于k位置说明能够跳到k位置的下一位，增加一个最大前后缀的长度 所以k+1之后作为next[i] ;
            if (p.charAt(i) == p.charAt(k)) {
                k++;//减一是因为这边k做下标用，和长度差1
            }
            next[i] = k;
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

    //316. 去除重复字母 单调栈
    public String removeDuplicateLetters(String s) {
//        ReentrantLock
        Stack<Character> stack = new Stack<>();
        boolean[] instack = new boolean[256];
        int[] count = new int[256];
        for (int i = 0; i < s.length(); i++) {
            count[s.charAt(i)]++;
        }
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            count[c]--;
            if (!instack[c]) {
                instack[c] = true;
                while (!stack.isEmpty() && stack.peek() > c && count[stack.peek()] > 0) {
                    instack[stack.pop()] = false;
                }
                stack.push(c);
            }
        }
        StringBuilder sb = new StringBuilder();
        stack.forEach(sb::append);
        return sb.toString();

    }


    public static void main(String[] args) {
        Strings obj = new Strings();
//        System.out.printf("" + obj.groupAnagrams(new String[]{"eat", "tea", "tan", "ate", "nat", "bat"}));
//        System.out.println(obj.reverseWords("the    sky   is   blue"));
//        System.out.println(obj.longestCommonPrefix(new String[]{"flower", "low", "flight"}));
//        System.out.println(obj.isPalindrome("ab_a"));
//        System.out.println(obj.removeDuplicateLetters("bcabc"));
//        System.out.println(obj.removeDuplicateLetters("bcac"));
//        obj.checkInclusion("adc", "dcda");
//        obj.minInsertions("(((()(()((())))(((()())))()())))(((()(()()((()()))");
//        obj.isSubsequence("abc", "cacbhbc");
//        System.out.println(obj.diffWaysToCompute("2*3-4*5"));
        System.out.println(obj.decodeString("abc3[cd]xyz"));

    }
}
