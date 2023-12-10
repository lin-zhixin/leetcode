package interview;

import javafx.util.Pair;

import java.lang.annotation.Target;
import java.util.*;
import java.util.stream.Collectors;

public class YT {
    //    依图笔试2023.8.31
    public static void main1(String[] args) {
        Scanner in = new Scanner(System.in);
        // 注意 hasNext 和 hasNextLine 的区别
        while (in.hasNextLine()) { // 注意 while 处理多个 case
            int n = Integer.parseInt(in.nextLine());
//            System.out.println(n);
            List<Book> books = new ArrayList<>();
            for (int i = 0; i < n; i++) {
//                while (in.hasNextLine()) {
                String[] book = in.nextLine().split(" ");
//                    System.out.println(book[0]+":"+book[1]);
                books.add(new Book(book[0], Integer.parseInt(book[1])));
//                }

            }
            books.sort(((o1, o2) -> o1.p - o2.p));
            books.forEach(e -> System.out.println(e.name));


//            int a = in.nextInt();
//            int b = in.nextInt();
//            System.out.println(a + b);
        }
    }

    public static void main2(String[] args) {
        Scanner in = new Scanner(System.in);
        // 注意 hasNext 和 hasNextLine 的区别
        while (in.hasNextLine()) { // 注意 while 处理多个 case
            int n = Integer.parseInt(in.nextLine());
            System.out.println("YES");
//            if (n == 0) {
//                System.out.println("YES");
//                continue;
//            }
//            int sum = 1;
//            List<Integer> list = new ArrayList<Integer>();
//            while (n > 0) {
//                sum *= n % 10;
//                list.add(n % 10);
//                n /= 10;
//            }
//            int target = 1;
//            boolean flage = false;
////            System.out.println(list);
//            out:
//            for (int i = 0; i < list.size(); i++) {
//                target *= list.get(i);
//                if (target == 0) {
//                    for (int j = i + 1; j < list.size(); j++) {
//                        if (list.get(j) == 0) {
//                            System.out.println("YES");
//                            flage = true;
//                            break out;
//                        }
//                    }
////                    System.out.println("NO");
//                    break out;
//                }
//                sum /= target;
//                if (target == sum) {
////                    System.out.println(integer);
//                    System.out.println("YES");
//                    flage = true;
//                    break;
//                }
//            }
//            if (!flage) System.out.println("NO");
//
//
////            int a = in.nextInt();
////            int b = in.nextInt();
////            System.out.println(a + b);
        }
    }

    public static void main(String[] args) {
        {
            Scanner in = new Scanner(System.in);
            // 注意 hasNext 和 hasNextLine 的区别
            while (in.hasNextLine()) { // 注意 while 处理多个 case
                String[] s = in.nextLine().split(" ");
                int n = Integer.parseInt(s[0]);
                int k = Integer.parseInt(s[1]);
                int d = Integer.parseInt(s[2]);
                int[] nums = Arrays.stream(in.nextLine().split(" ")).mapToInt(Integer::parseInt).toArray();
                int[] dp = new int[n];
                int maxres = 0;
                Map<Integer, PriorityQueue<Integer>> heap = new HashMap<>();
                dp[0] = nums[0];
                heap.put(0, new PriorityQueue<>());
                heap.get(0).offer(nums[0]);
                for (int i = 1; i < k; i++) {
                    dp[i] = dp[i - 1] * nums[i];
                    heap.put(i, new PriorityQueue<>(heap.get(i - 1)));
                    heap.get(i).offer(nums[i]);
                }
                for (int i = k; i < n; i++) {
                    int maxind = i - 1;
                    for (int j = 1; j < d; j++) {
                        System.out.println(i - j);
                        if (nums[i] > heap.get(i - j).peek() && nums[i] * dp[i - j] / heap.get(i - j).peek() > dp[i - j]) {
                            maxind = i - j ;
                        }
                    }
                    dp[i] = (dp[maxind] / heap.get(maxind).peek()) * nums[i];
                    PriorityQueue<Integer> t = new PriorityQueue<>(heap.get(maxind));
                    t.poll();
                    t.offer(nums[i]);
                    heap.put(i, t);
                    maxres = Math.max(maxres, dp[i]);
                }
                heap.forEach((key,v)->{
                    System.out.println(key+":"+v);
                });
                System.out.println(maxres);
            }
        }
    }
}

class Book {
    String name;
    Integer p;

    public Book() {
    }

    public Book(String name, Integer p) {
        this.name = name;
        this.p = p;
    }

    @Override
    public String toString() {
        return "Book{" +
                "name='" + name + '\'' +
                ", p=" + p +
                '}';
    }
}

