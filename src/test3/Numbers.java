package test3;

//数字相关题目

import category.MyUtile;
import category.SortTest;
import javafx.util.Pair;

import java.awt.image.WritableRenderedImage;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;
//import org.apache.commons.collections.CollectionUtils;

public class Numbers {
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
        System.out.println(nu.minWindow("ADOBECODEBANC", "ABC"));

    }
}





